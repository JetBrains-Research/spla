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

#include <algo/vector/SplaVectorReadDense.hpp>
#include <core/SplaLibraryPrivate.hpp>
#include <core/SplaQueueFinisher.hpp>
#include <storage/block/SplaVectorDense.hpp>

bool spla::VectorReadDense::Select(const spla::AlgorithmParams &params) const {
    auto p = dynamic_cast<const ParamsVectorRead *>(&params);

    return p &&
           p->v.Is<VectorDense>();
}

void spla::VectorReadDense::Process(spla::AlgorithmParams &params) {
    using namespace boost;

    auto p = dynamic_cast<ParamsVectorRead *>(&params);
    auto library = p->desc->GetLibrary().GetPrivatePtr();

    auto device = library->GetDeviceManager().GetDevice(p->deviceId);
    compute::context ctx = library->GetContext();
    compute::command_queue queue(ctx, device);
    QueueFinisher finisher(queue, library->GetLogger());

    auto v = p->v.Cast<VectorDense>();
    auto d = p->d;
    auto byteSize = p->byteSize;

    auto hostRows = d->GetRows();
    auto hostVals = reinterpret_cast<unsigned char *>(d->GetVals());

    assert(hostRows || hostVals);

    SPDLOG_LOGGER_TRACE(library->GetLogger(), "Copy vector size={} storage nvals={} offset={}",
                        v->GetNrows(), v->GetNvals(), p->offset);

    std::vector<Index> copiedMask(v->GetMask().size());
    compute::copy(v->GetMask().begin(), v->GetMask().end(), copiedMask.begin(), queue);

    if (hostRows) {
        std::size_t k = 0;
        for (std::size_t i = 0; i < copiedMask.size(); i++) {
            if (copiedMask[i]) {
                hostRows[p->offset + k] = p->baseI + i;
                k += 1;
            }
        }
    }

    if (hostVals && byteSize) {
        std::vector<unsigned char> copiedValues(v->GetVals().size());
        compute::copy(v->GetVals().begin(), v->GetVals().end(), copiedValues.begin(), queue);

        std::size_t k = 0;
        for (std::size_t i = 0; i < copiedMask.size(); i++) {
            if (copiedMask[i]) {
                for (std::size_t j = 0; j < byteSize; j++) {
                    hostVals[p->offset * byteSize + k * byteSize + j] = copiedValues[i * byteSize + j];
                }
                k += 1;
            }
        }
    }
}

spla::Algorithm::Type spla::VectorReadDense::GetType() const {
    return spla::Algorithm::Type::VectorRead;
}

std::string spla::VectorReadDense::GetName() const {
    return "VectorReadDense";
}
