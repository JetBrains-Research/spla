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

#include <spla-cpp/SplaScalar.hpp>
#include <storage/SplaScalarStorage.hpp>

bool spla::Scalar::HasValue() const {
    return mStorage->HasValue();
}

const spla::RefPtr<spla::ScalarStorage> &spla::Scalar::GetStorage() const {
    return mStorage;
}

void spla::Scalar::Dump(std::ostream &stream) const {
    mStorage->Dump(stream);
}

spla::RefPtr<spla::Object> spla::Scalar::Clone() const {
    return RefPtr<Scalar>(new Scalar(GetType(), GetLibrary(), GetStorage()->Clone())).As<Object>();
}

spla::RefPtr<spla::Scalar> spla::Scalar::Make(const spla::RefPtr<spla::Type> &type, spla::Library &library) {
    return RefPtr<spla::Scalar>(new Scalar(type, library));
}

spla::Scalar::Scalar(const spla::RefPtr<spla::Type> &type, spla::Library &library, RefPtr<class ScalarStorage> storage)
    : TypedObject(type, TypeName::Scalar, library) {
    mStorage = storage.IsNotNull() ? std::move(storage) : ScalarStorage::Make(GetLibrary());
}

spla::RefPtr<spla::Object> spla::Scalar::CloneEmpty() {
    return Make(GetType(), GetLibrary()).As<Object>();
}

void spla::Scalar::CopyData(const spla::RefPtr<spla::Object> &object) {
    auto scalar = dynamic_cast<const Scalar *>(object.Get());
    assert(scalar);

    if (scalar) {
        assert(GetType() == scalar->GetType());
        mStorage = scalar->GetStorage();
    }
}

spla::Scalar::~Scalar() = default;
