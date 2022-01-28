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

#include <algo/matrix/SplaMatrixTransposeCOO.hpp>
#include <compute/SplaApplyMask.hpp>
#include <compute/SplaMaskByKey.hpp>
#include <compute/SplaSortByRowColumn.hpp>
#include <core/SplaLibraryPrivate.hpp>
#include <core/SplaQueueFinisher.hpp>
#include <storage/SplaMatrixStorage.hpp>
#include <storage/block/SplaMatrixCOO.hpp>

bool spla::MatrixTransposeCOO::Select(const spla::AlgorithmParams &params) const {
    auto p = dynamic_cast<const ParamsTranspose *>(&params);

    return p &&
           p->mask.Is<MatrixCOO>() &&
           p->a.Is<MatrixCOO>();
}

void spla::MatrixTransposeCOO::Process(spla::AlgorithmParams &params) {
    using namespace boost;

    auto p = dynamic_cast<ParamsTranspose *>(&params);
    auto library = p->desc->GetLibrary().GetPrivatePtr();
    auto &desc = p->desc;

    auto device = library->GetDeviceManager().GetDevice(p->deviceId);
    compute::context ctx = library->GetContext();
    compute::command_queue queue(ctx, device);
    QueueFinisher finisher(queue, library->GetLogger());

    auto &type = p->type;
    auto byteSize = type->GetByteSize();
    auto typeHasValues = type->HasValues();

    auto a = p->a.Cast<MatrixCOO>();
    auto mask = p->mask.Cast<MatrixCOO>();
    auto complementMask = desc->IsParamSet(Descriptor::Param::MaskComplement);

    // Do not process empty block
    if (a.IsNull())
        return;
    // Nothing to do
    if (p->hasMask && !complementMask && mask.IsNull())
        return;

    // Buffers to store result
    compute::vector<unsigned int> rows(a->GetNvals(), ctx);
    compute::vector<unsigned int> cols(a->GetNvals(), ctx);
    compute::vector<unsigned char> vals(ctx);

    // Copy indices
    compute::copy(a->GetRows().begin(), a->GetRows().end(), cols.begin(), queue);
    compute::copy(a->GetCols().begin(), a->GetCols().end(), rows.begin(), queue);

    // If has values, copy values
    if (typeHasValues) {
        vals.resize(a->GetNvals() * byteSize, queue);
        compute::copy(a->GetVals().begin(), a->GetVals().end(), vals.begin(), queue);
    }

    // Sort to reorder indices
    SortByRowColumn(rows, cols, vals, byteSize, queue);

    // Apply finally mask if required
    if (p->hasMask && mask.IsNotNull()) {
        compute::vector<unsigned int> tmpRows(ctx);
        compute::vector<unsigned int> tmpCols(ctx);
        compute::vector<unsigned char> tmpVals(ctx);

        ApplyMask(mask->GetRows(), mask->GetCols(), rows, cols, vals, tmpRows, tmpCols, tmpVals, byteSize, complementMask, queue);

        std::swap(rows, tmpRows);
        std::swap(cols, tmpCols);
        std::swap(vals, tmpVals);
    }

    // Save if result is not empty
    if (!rows.empty()) {
        auto nrowsT = a->GetNcols();
        auto ncolsT = a->GetNrows();
        auto nvalsT = rows.size();
        p->w = MatrixCOO::Make(nrowsT, ncolsT, nvalsT, std::move(rows), std::move(cols), std::move(vals)).As<MatrixBlock>();
    }
}

spla::Algorithm::Type spla::MatrixTransposeCOO::GetType() const {
    return spla::Algorithm::Type::Transpose;
}

std::string spla::MatrixTransposeCOO::GetName() const {
    return "TransposeCOO";
}
