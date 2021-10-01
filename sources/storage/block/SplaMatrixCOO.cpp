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

#include <storage/block/SplaMatrixCOO.hpp>

spla::RefPtr<spla::MatrixCOO> spla::MatrixCOO::Make(size_t nrows, size_t ncols, size_t nvals,
                                                    spla::RefPtr<spla::Svm<unsigned int>> rows,
                                                    spla::RefPtr<spla::Svm<unsigned int>> cols,
                                                    spla::RefPtr<spla::Svm<unsigned char>> vals) {
    return spla::RefPtr<spla::MatrixCOO>(new MatrixCOO(nrows, ncols, nvals, std::move(rows), std::move(cols), std::move(vals)));
}

spla::MatrixCOO::MatrixCOO(size_t nrows, size_t ncols, size_t nvals,
                           spla::RefPtr<spla::Svm<unsigned int>> rows,
                           spla::RefPtr<spla::Svm<unsigned int>> cols,
                           spla::RefPtr<spla::Svm<unsigned char>> vals)
    : MatrixBlock(nrows, ncols, nvals),
      mRows(std::move(rows)),
      mCols(std::move(cols)),
      mVals(std::move(vals)) {
}