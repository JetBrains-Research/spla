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
#include <core/SplaMath.hpp>
#include <storage/SplaVectorStorage.hpp>

void spla::VectorStorage::Clear() {
    std::lock_guard<std::mutex> lock(mMutex);
    mNvals = 0;
    for (auto &b : mBlocks) b.clear();
}

void spla::VectorStorage::SetBlock(const spla::VectorStorage::Index &index, const spla::RefPtr<spla::VectorBlock> &block) {
    assert(index < mNblockRows);
    assert(block.IsNotNull());

    std::lock_guard<std::mutex> lock(mMutex);
    auto &prev = mBlocks.front()[index];

    if (prev.IsNotNull())
        mNvals -= prev->GetNvals();

    prev = block;
    mNvals += block->GetNvals();
}

void spla::VectorStorage::SetBlock(const Index &index, const RefPtr<VectorBlock> &block, boost::compute::command_queue &queue) {
    assert(index < mNblockRows);
    assert(block.IsNotNull());

    std::lock_guard<std::mutex> lock(mMutex);
    auto &prev = mBlocks.front()[index];

    if (prev.IsNotNull())
        mNvals -= prev->GetNvals();

    prev = block;
    mNvals += block->GetNvals();

    for (std::size_t i = 1; i < mBlocks.size(); i++) mBlocks[i][index] = block->Clone(queue);
}

void spla::VectorStorage::SetBlock(const Index &index, const RefPtr<VectorBlock> &block, std::size_t deviceId) {
    using namespace boost;
    compute::device device = mLibrary.GetPrivate().GetDeviceManager().GetDevice(deviceId);
    compute::command_queue queue(mLibrary.GetPrivate().GetContext(), device);
    SetBlock(index, block, queue);
}

void spla::VectorStorage::GetBlocks(spla::VectorStorage::EntryList &entryList) const {
    std::lock_guard<std::mutex> lock(mMutex);
    entryList.clear();
    entryList.insert(entryList.begin(), mBlocks.front().begin(), mBlocks.front().end());
}

void spla::VectorStorage::GetBlocks(EntryMap &entryMap) const {
    std::lock_guard<std::mutex> lock(mMutex);
    entryMap = mBlocks.front();
}

void spla::VectorStorage::GetBlocksGrid(std::size_t &rows) const {
    rows = mNblockRows;
}

spla::RefPtr<spla::VectorBlock> spla::VectorStorage::GetBlock(const spla::VectorStorage::Index &index, std::size_t deviceId) const {
    assert(index < mNblockRows);
    assert(deviceId < mBlocks.size());
    std::lock_guard<std::mutex> lock(mMutex);
    auto entry = mBlocks[deviceId].find(index);
    return entry != mBlocks[deviceId].end() ? entry->second : nullptr;
}

std::size_t spla::VectorStorage::GetNrows() const noexcept {
    return mNrows;
}

std::size_t spla::VectorStorage::GetNvals() const noexcept {
    std::lock_guard<std::mutex> lock(mMutex);
    return mNvals;
}

spla::RefPtr<spla::VectorStorage> spla::VectorStorage::Make(std::size_t nrows, spla::Library &library) {
    return {new VectorStorage(nrows, library)};
}

spla::VectorStorage::VectorStorage(std::size_t nrows, spla::Library &library)
    : mNrows(nrows), mLibrary(library) {
    mBlockSize = mLibrary.GetPrivate().GetBlockSize();
    mNblockRows = math::GetBlocksCount(nrows, mBlockSize);
    mBlocks.resize(library.GetPrivate().GetDeviceManager().GetDevicesCount());
}

void spla::VectorStorage::RemoveBlock(const spla::VectorStorage::Index &index) {
    assert(index < mNblockRows);

    std::lock_guard<std::mutex> lock(mMutex);
    for (auto &b : mBlocks) b.erase(index);
}

std::size_t spla::VectorStorage::GetNblockRows() const noexcept {
    return mNblockRows;
}

std::size_t spla::VectorStorage::GetBlockSize() const noexcept {
    return mBlockSize;
}

void spla::VectorStorage::Dump(std::ostream &stream) const {
    std::lock_guard<std::mutex> lock(mMutex);

    stream << "VectorStorage:"
           << " nrows=" << mNrows
           << " nvals=" << mNvals
           << " bcount=" << mBlocks.size()
           << " bsize=" << mBlockSize << std::endl;

    auto bsize = static_cast<unsigned int>(mBlockSize);

    for (auto &entry : mBlocks.front()) {
        auto index = entry.first;
        auto &block = entry.second;
        stream << "Block (" << index << ") ";
        block->Dump(stream, index * bsize);
    }
}

spla::RefPtr<spla::VectorStorage> spla::VectorStorage::Clone() const {
    std::lock_guard<std::mutex> lock(mMutex);

    auto storage = Make(GetNrows(), mLibrary);
    storage->mBlockSize = mBlockSize;
    storage->mBlocks = mBlocks;
    storage->mNvals = mNvals;

    return storage;
}
