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

#ifndef SPLA_SPLAEXPRESSIONMANAGER_HPP
#define SPLA_SPLAEXPRESSIONMANAGER_HPP

#include <expression/SplaNodeProcessor.hpp>
#include <spla-cpp/SplaExpression.hpp>
#include <spla-cpp/SplaExpressionNode.hpp>
#include <spla-cpp/SplaLibrary.hpp>
#include <unordered_map>

namespace spla {

    /**
     * @addtogroup Internal
     * @{
     */

    /**
     * @brief ExpressionManager
     *
     * Allows to submit for execution provided expressions.
     * Checks expression correctness, defines traversal order,
     * finds and calls actual node processors to process nodes.
     *
     * Provides functionality to register custom processors for
     * desired expression node operations.
     *
     * @see Expression
     * @see ExpressionNode
     * @see NodeProcessor
     */
    class ExpressionManager final : public RefCnt {
    public:
        /** Expression DAG structure */
        struct TraversalInfo {
            /** Expression */
            RefPtr<Expression> expression;
            /** Nodes to start evaluation */
            std::vector<std::size_t> startNodes;
            /** Final nodes */
            std::vector<std::size_t> endNodes;
        };

        explicit ExpressionManager(Library &library);
        ~ExpressionManager() override = default;

        void Submit(const RefPtr<Expression> &expression);
        void Register(const RefPtr<NodeProcessor> &processor);

    private:
        void FindStartNodes(TraversalInfo &context);
        void FindEndNodes(TraversalInfo &context);
        void CheckCycles(TraversalInfo &context);
        bool CheckCyclesImpl(std::size_t idx, std::vector<int> &visited, const std::vector<RefPtr<ExpressionNode>> &nodes);

        RefPtr<NodeProcessor> SelectProcessor(std::size_t nodeIdx, const Expression &expression);

    private:
        using ProcessorList = std::vector<RefPtr<NodeProcessor>>;
        using ProcessorMap = std::unordered_map<ExpressionNode::Operation, ProcessorList>;

        ProcessorMap mProcessors;
        Library &mLibrary;
    };

    /**
     * @}
     */

}// namespace spla

#endif//SPLA_SPLAEXPRESSIONMANAGER_HPP
