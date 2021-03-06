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

#include <Testing.hpp>

void testCommon(spla::Library &library, std::size_t M, std::size_t N, std::size_t nvals, std::size_t seed = 0) {
    utils::Matrix source = utils::Matrix<float>::Generate(M, N, nvals, seed);
    source.Fill(utils::UniformRealGenerator<float>());

    auto spT = spla::Types::Float32(library);
    auto spM = spla::Matrix::Make(M, N, spT, library);
    auto spDesc = spla::Descriptor::Make(library);

    auto spDataSrc = spla::DataMatrix::Make(library);
    spDataSrc->SetRows(source.GetRows());
    spDataSrc->SetCols(source.GetCols());
    spDataSrc->SetVals(source.GetVals());
    spDataSrc->SetNvals(source.GetNvals());

    auto spExprWrite = spla::Expression::Make(library);
    auto spWrite = spExprWrite->MakeDataWrite(spM, spDataSrc, spDesc);
    spExprWrite->Submit();
    spExprWrite->Wait();
    ASSERT_EQ(spExprWrite->GetState(), spla::Expression::State::Evaluated);

    utils::Matrix<float> expected = source.SortReduceDuplicates();
    ASSERT_TRUE(expected.Equals(spM));
}

void testSortedNoDuplicates(spla::Library &library, std::size_t M, std::size_t N, std::size_t nvals, std::size_t seed = 0) {
    utils::Matrix source = utils::Matrix<float>::Generate(M, N, nvals, seed).SortReduceDuplicates();
    source.Fill(utils::UniformRealGenerator<float>());

    auto spT = spla::Types::Float32(library);
    auto spM = spla::Matrix::Make(M, N, spT, library);

    // Specify, that values already in row-col order + no duplicates
    auto spDesc = spla::Descriptor::Make(library);
    spDesc->SetParam(spla::Descriptor::Param::ValuesSorted);
    spDesc->SetParam(spla::Descriptor::Param::NoDuplicates);

    auto spDataSrc = spla::DataMatrix::Make(library);
    spDataSrc->SetRows(source.GetRows());
    spDataSrc->SetCols(source.GetCols());
    spDataSrc->SetVals(source.GetVals());
    spDataSrc->SetNvals(source.GetNvals());

    auto spExprWrite = spla::Expression::Make(library);
    auto spWrite = spExprWrite->MakeDataWrite(spM, spDataSrc, spDesc);
    spExprWrite->Submit();
    spExprWrite->Wait();
    ASSERT_EQ(spExprWrite->GetState(), spla::Expression::State::Evaluated);

    utils::Matrix<float> &expected = source;
    ASSERT_TRUE(expected.Equals(spM));
}

void test(std::size_t M, std::size_t N, std::size_t base, std::size_t step, std::size_t iter) {
    utils::testBlocks({1000, 10000, 100000}, [=](spla::Library &library) {
        for (std::size_t i = 0; i < iter; i++) {
            std::size_t nvals = base + i * step;
            testCommon(library, M, N, nvals, i);
        }

        for (std::size_t i = 0; i < iter; i++) {
            std::size_t nvals = base + i * step;
            testSortedNoDuplicates(library, M, N, nvals, i);
        }
    });
}

TEST(DataMatrix, Small) {
    std::size_t M = 100, N = 200;
    test(M, N, M, M, 10);
}

TEST(DataMatrix, Medium) {
    std::size_t M = 1300, N = 2100;
    test(M, N, M, M, 10);
}

TEST(DataMatrix, Large) {
    std::size_t M = 10300, N = 18000;
    test(M, N, M, M, 5);
}

SPLA_GTEST_MAIN