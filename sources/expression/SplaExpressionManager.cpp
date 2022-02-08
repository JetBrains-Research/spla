/**********************************************************************************/
/* This file is part of spla project                                              */
/* https://github.com/JetBrains-Research/spla                                     */
/**********************************************************************************/
/* MIT License                                                                    */
/*                                                                                */
/* Copyright (c) 2021 JetBrains-Research                                          */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files (the "Software"), to deal  */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

#include <algorithm>
#include <vector>

#include <core/SplaError.hpp>
#include <core/SplaLibraryPrivate.hpp>
#include <expression/SplaExpressionFuture.hpp>
#include <expression/SplaExpressionTasks.hpp>

#include <expression/matrix/SplaMatrixDataRead.hpp>
#include <expression/matrix/SplaMatrixDataWrite.hpp>
#include <expression/matrix/SplaMatrixEWiseAdd.hpp>
#include <expression/matrix/SplaMatrixReduceScalar.hpp>
#include <expression/matrix/SplaMatrixTranspose.hpp>
#include <expression/matrix/SplaMatrixTril.hpp>
#include <expression/matrix/SplaMatrixTriu.hpp>

#include <expression/prod/SplaMxM.hpp>
#include <expression/prod/SplaVxM.hpp>

#include <expression/scalar/SplaScalarDataRead.hpp>
#include <expression/scalar/SplaScalarDataWrite.hpp>
#include <expression/scalar/SplaScalarEWiseAdd.hpp>

#include <expression/vector/SplaVectorAssign.hpp>
#include <expression/vector/SplaVectorDataRead.hpp>
#include <expression/vector/SplaVectorDataWrite.hpp>
#include <expression/vector/SplaVectorEWiseAdd.hpp>
#include <expression/vector/SplaVectorReduce.hpp>
#include <expression/vector/SplaVectorToDense.hpp>

spla::ExpressionManager::ExpressionManager(spla::Library &library) : mLibrary(library) {
    // Here we can register built-in processors, one by on
    // NOTE: order of registration matters. First one has priority among others for the same op.
    Register(new MatrixDataRead());
    Register(new MatrixDataWrite());
    Register(new MatrixEWiseAdd());
    Register(new MatrixTranspose());
    Register(new MatrixTril());
    Register(new MatrixTriu());
    Register(new MatrixReduceScalar());
    Register(new ScalarDataRead());
    Register(new ScalarDataWrite());
    Register(new ScalarEWiseAdd());
    Register(new VectorToDense());
    Register(new VectorAssign());
    Register(new VectorDataWrite());
    Register(new VectorDataRead());
    Register(new VectorEWiseAdd());
    Register(new VectorReduce());
    Register(new MxM());
    Register(new VxM());
}

void spla::ExpressionManager::Submit(const spla::RefPtr<spla::Expression> &expression) {
    CHECK_RAISE_ERROR(expression.IsNotNull(), InvalidArgument, "Passed null expression");
    CHECK_RAISE_ERROR(expression->GetState() == Expression::State::Default, InvalidArgument,
                      "Passed expression=" << expression->GetLabel() << " must be in `Default` state before evaluation");

    expression->SetState(Expression::State::Submitted);

    // NOTE: Special case, expression without nodes
    // Nothing to do here, return
    if (expression->Empty()) {
        expression->SetState(Expression::State::Evaluated);
        return;
    }

    TraversalInfo context;
    context.expression = expression;

    FindStartNodes(context);
    CHECK_RAISE_ERROR(!context.startNodes.empty(), InvalidArgument,
                      "No start nodes to run computation in expression=" << expression->GetLabel()
                                                                         << "; Possibly have some dependency cycle?");

    FindEndNodes(context);
    CHECK_RAISE_ERROR(!context.endNodes.empty(), InvalidArgument,
                      "No end nodes in expression=" << expression->GetLabel()
                                                    << "; Possibly have some dependency cycle?");

    CheckCycles(context);

    auto &nodes = expression->GetNodes();
    auto expressionTasks = std::make_unique<ExpressionTasks>();
    auto &taskflow = expressionTasks->taskflow;

    std::vector<tf::Task> modules;
    modules.reserve(nodes.size());

    for (std::size_t idx = 0; idx < nodes.size(); idx++) {
        // Select processor for node
        auto processor = SelectProcessor(idx, *expression);
        // Wrap processor into task to handle dynamic changes of expression nodes params
        auto task = [idx, expression = expression.Get(), processor = processor.Get()](tf::Subflow &subflow) {
            // If aborted in previous tasks, candle run
            if (expression->GetState() == Expression::State::Aborted)
                return;

            // Check, if out arg appears in the input list
            auto &args = expression->GetNodes()[idx]->GetArgs();
            if (!args.empty()) {
                auto out = args[0];
                auto query = std::find(args.begin() + 1, args.end(), out);
                // If appears, must create a proxy
                // And replace all `in` entries with read-only proxy
                if (query != args.end()) {
                    auto proxy = out->Clone();
                    std::for_each(args.begin() + 1, args.end(), [&](RefPtr<Object> &arg) {
                        if (arg == out)
                            arg = proxy;
                    });
                }
            }

            // Task build wraps error handling and abortion
            TaskBuilder taskBuilder(expression, subflow);

            try {
                // Actual task graph composition
                processor->Process(idx, *expression, taskBuilder);
            } catch (std::exception &ex) {
                expression->SetState(Expression::State::Aborted);
                auto logger = expression->GetLibrary().GetPrivate().GetLogger();
                SPDLOG_LOGGER_ERROR(logger, "Error inside expression node. {}", ex.what());
            }
        };
        modules.push_back(taskflow.emplace(std::move(task)));
    }

    for (std::size_t idx = 0; idx < nodes.size(); idx++) {
        // Compose final taskflow graph
        auto &node = nodes[idx];
        auto &next = node->GetNext();
        auto &thisModule = modules[idx];

        // For each child make thisModule -> nextModule dependency
        for (auto &n : next) {
            auto &nextModule = modules[n->GetIdx()];
            thisModule.precede(nextModule);
        }
    }

    // Dummy task to notify expression state
    auto notification = taskflow.emplace([expression = expression.Get()]() {
                                    if (expression->GetState() != Expression::State::Aborted)
                                        expression->SetState(Expression::State::Evaluated);
                                })
                                .name("Notification");

    // Add dependency for end nodes
    for (auto end : context.endNodes)
        modules[end].precede(notification);

    auto &executor = mLibrary.GetPrivate().GetTaskFlowExecutor();
    auto expressionFuture = std::make_unique<ExpressionFuture>(executor.run(taskflow));

    expression->SetTasks(std::move(expressionTasks));
    expression->SetFuture(std::move(expressionFuture));
}

void spla::ExpressionManager::Register(const spla::RefPtr<spla::NodeProcessor> &processor) {
    CHECK_RAISE_ERROR(processor.IsNotNull(), InvalidArgument, "Passed null processor");

    ExpressionNode::Operation op = processor->GetOperationType();
    auto list = mProcessors.find(op);

    if (list == mProcessors.end())
        list = mProcessors.emplace(op, ProcessorList()).first;

    list->second.push_back(processor);
}

void spla::ExpressionManager::FindStartNodes(spla::ExpressionManager::TraversalInfo &context) {
    auto &expression = context.expression;
    auto &nodes = expression->GetNodes();
    auto &start = context.startNodes;

    for (std::size_t idx = 0; idx < nodes.size(); idx++) {
        // No incoming dependency
        if (nodes[idx]->GetPrev().empty())
            start.push_back(idx);
    }
}

void spla::ExpressionManager::FindEndNodes(spla::ExpressionManager::TraversalInfo &context) {
    auto &expression = context.expression;
    auto &nodes = expression->GetNodes();
    auto &end = context.endNodes;

    for (std::size_t idx = 0; idx < nodes.size(); idx++) {
        // No incoming dependency
        if (nodes[idx]->GetNext().empty())
            end.push_back(idx);
    }
}

void spla::ExpressionManager::CheckCycles(spla::ExpressionManager::TraversalInfo &context) {
    auto &expression = context.expression;
    auto &nodes = expression->GetNodes();
    auto &start = context.startNodes;
    auto nodesCount = nodes.size();

    for (auto n : start) {
        std::vector<int> visited(nodesCount, 0);
        auto cycle = CheckCyclesImpl(n, visited, nodes);
        CHECK_RAISE_ERROR(!cycle, InvalidArgument, "Provided expression has dependency cycle");
    }
}

bool spla::ExpressionManager::CheckCyclesImpl(std::size_t idx, std::vector<int> &visited,
                                              const std::vector<RefPtr<ExpressionNode>> &nodes) {
    if (visited[idx])
        return true;

    auto &node = nodes[idx];
    auto &next = node->GetNext();

    for (const auto &n : next) {
        if (CheckCyclesImpl(n->GetIdx(), visited, nodes))
            return true;
    }

    return false;
}

spla::RefPtr<spla::NodeProcessor>
spla::ExpressionManager::SelectProcessor(std::size_t nodeIdx, const spla::Expression &expression) {
    auto &nodes = expression.GetNodes();
    auto &node = nodes[nodeIdx];
    ExpressionNode::Operation op = node->GetNodeOp();

    auto iter = mProcessors.find(op);

    CHECK_RAISE_ERROR(iter != mProcessors.end(), InvalidState,
                      "No processors for such op=" << ExpressionNodeOpToStr(op));

    const auto &processors = iter->second;

    // NOTE: Iterate through all processors for this operation and
    // select the first one, which meets requirements
    for (auto &processor : processors)
        if (processor->Select(nodeIdx, expression))
            return processor;

    RAISE_ERROR(InvalidState,
                "Failed to find suitable processor for the node op=" << ExpressionNodeOpToStr(op));
}