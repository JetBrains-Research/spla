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

#include <spla-cpp/SplaData.hpp>
#include <spla-cpp/SplaLibrary.hpp>

spla::Data::Data(spla::Object::TypeName dataType, spla::Library &library) : Object(dataType, library) {
}

void spla::Data::SetReleaseProc(ReleaseProc releaseProc) {
    mReleaseProc = std::move(releaseProc);
}

spla::DataMatrix::DataMatrix(spla::Library &library) : Data(Object::TypeName::DataMatrix, library) {
}

spla::DataMatrix::~DataMatrix() {
    if (mReleaseProc) {
        if (mRows)
            mReleaseProc(mRows);
        if (mCols)
            mReleaseProc(mCols);
        if (mValues)
            mReleaseProc(mValues);

        mRows = nullptr;
        mCols = nullptr;
        mValues = nullptr;
    }
}

void spla::DataMatrix::SetRows(unsigned int *rows) {
    mRows = rows;
}

void spla::DataMatrix::SetCols(unsigned int *cols) {
    mCols = cols;
}

void spla::DataMatrix::SetVals(void *values) {
    mValues = values;
}

unsigned int *spla::DataMatrix::GetRows() const {
    return mRows;
}

unsigned int *spla::DataMatrix::GetCols() const {
    return mCols;
}

void *spla::DataMatrix::GetVals() const {
    return mValues;
}

void spla::DataMatrix::SetNvals(std::size_t nvals) {
    mNvals = nvals;
}

std::size_t spla::DataMatrix::GetNvals() const {
    return mNvals;
}

spla::RefPtr<spla::DataMatrix> spla::DataMatrix::Make(spla::Library &library) {
    return spla::RefPtr<spla::DataMatrix>(new DataMatrix(library));
}

spla::RefPtr<spla::DataMatrix> spla::DataMatrix::Make(unsigned int *rows, unsigned int *cols, void *values, std::size_t nvals, Library &library) {
    auto data = Make(library);
    data->SetRows(rows);
    data->SetCols(cols);
    data->SetVals(values);
    data->SetNvals(nvals);
    return data;
}

spla::DataVector::DataVector(spla::Library &library) : Data(Object::TypeName::DataVector, library) {
}

spla::DataVector::~DataVector() {
    if (mReleaseProc) {
        if (mRows)
            mReleaseProc(mRows);
        if (mValues)
            mReleaseProc(mValues);

        mRows = nullptr;
        mValues = nullptr;
    }
}

void spla::DataVector::SetRows(unsigned int *rows) {
    mRows = rows;
}

void spla::DataVector::SetVals(void *values) {
    mValues = values;
}

unsigned int *spla::DataVector::GetRows() const {
    return mRows;
}


void *spla::DataVector::GetVals() const {
    return mValues;
}

void spla::DataVector::SetNvals(std::size_t nvals) {
    mNvals = nvals;
}

std::size_t spla::DataVector::GetNvals() const {
    return mNvals;
}

spla::RefPtr<spla::DataVector> spla::DataVector::Make(spla::Library &library) {
    return spla::RefPtr<spla::DataVector>(new DataVector(library));
}

spla::RefPtr<spla::DataVector> spla::DataVector::Make(unsigned int *rows, void *values, std::size_t nvals, Library &library) {
    auto data = Make(library);
    data->SetRows(rows);
    data->SetVals(values);
    data->SetNvals(nvals);
    return data;
}

spla::DataScalar::DataScalar(spla::Library &library) : Data(Object::TypeName::DataScalar, library) {
}

spla::DataScalar::~DataScalar() {
    if (mReleaseProc) {
        if (mValue)
            mReleaseProc(mValue);

        mValue = nullptr;
    }
}

void spla::DataScalar::SetValue(void *value) {
    mValue = value;
}

void *spla::DataScalar::GetValue() const {
    return mValue;
}

spla::RefPtr<spla::DataScalar> spla::DataScalar::Make(spla::Library &library) {
    return spla::RefPtr<spla::DataScalar>(new DataScalar(library));
}

spla::RefPtr<spla::DataScalar> spla::DataScalar::Make(void *value, spla::Library &library) {
    auto data = Make(library);
    data->SetValue(value);
    return data;
}