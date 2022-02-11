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

#include <core/SplaDeviceManager.hpp>
#include <core/SplaLibraryPrivate.hpp>
#include <storage/SplaScalarStorage.hpp>
#include <storage/SplaScalarValue.hpp>

spla::ScalarStorage::~ScalarStorage() = default;

void spla::ScalarStorage::SetValue(const RefPtr<ScalarValue> &value) {
    assert(value.IsNotNull());

    std::lock_guard<std::mutex> lock(mMutex);
    mValues.front() = value;
}

void spla::ScalarStorage::SetValue(const RefPtr<class ScalarValue> &value, boost::compute::command_queue &queue) {
    assert(value.IsNotNull());
    std::lock_guard<std::mutex> lock(mMutex);

    mValues.front() = value;
    for (std::size_t i = 1; i < mValues.size(); i++) mValues[i] = value->Clone(queue);
}

void spla::ScalarStorage::RemoveValue() {
    std::lock_guard<std::mutex> lock(mMutex);
    for (auto &v : mValues) v.Reset();
}

bool spla::ScalarStorage::HasValue() const {
    std::lock_guard<std::mutex> lock(mMutex);
    return mValues.front().IsNotNull();
}

spla::RefPtr<spla::ScalarValue> spla::ScalarStorage::GetValue(std::size_t deviceId) const {
    std::lock_guard<std::mutex> lock(mMutex);
    assert(deviceId < mValues.size());
    return mValues[deviceId];
}

void spla::ScalarStorage::Dump(std::ostream &stream) const {
    std::lock_guard<std::mutex> lock(mMutex);
    if (mValues.front().IsNotNull()) mValues.front()->Dump(stream);
}

spla::RefPtr<spla::ScalarStorage> spla::ScalarStorage::Clone() const {
    std::lock_guard<std::mutex> lock(mMutex);

    auto storage = Make(mLibrary);
    storage->mValues = mValues;

    return storage;
}

spla::RefPtr<spla::ScalarStorage> spla::ScalarStorage::Make(spla::Library &library) {
    return spla::RefPtr<spla::ScalarStorage>(new ScalarStorage(library));
}

spla::ScalarStorage::ScalarStorage(spla::Library &library) : mLibrary(library) {
    mValues.resize(library.GetPrivate().GetDeviceManager().GetDevicesCount());
}