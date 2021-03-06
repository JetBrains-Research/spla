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

void testCommon(spla::Library &library, std::size_t M, std::size_t nvals, std::size_t seed = 0) {
    utils::Vector source = utils::Vector<float>::Generate(M, nvals, seed);
    source.Fill(utils::UniformRealGenerator<float>());

    auto spT = spla::Types::Float32(library);
    auto spV = spla::Vector::Make(M, spT, library);
    auto spDesc = spla::Descriptor::Make(library);

    auto spDataSrc = spla::DataVector::Make(library);
    spDataSrc->SetRows(source.GetRows());
    spDataSrc->SetVals(source.GetVals());
    spDataSrc->SetNvals(source.GetNvals());

    auto spExprWrite = spla::Expression::Make(library);
    auto spWrite = spExprWrite->MakeDataWrite(spV, spDataSrc, spDesc);
    spExprWrite->Submit();
    spExprWrite->Wait();
    ASSERT_EQ(spExprWrite->GetState(), spla::Expression::State::Evaluated);

    utils::Vector expected = source.SortReduceDuplicates();

    ASSERT_TRUE(expected.Equals(spV));
}

void testSortedNoDuplicates(spla::Library &library, std::size_t M, std::size_t nvals, std::size_t seed = 0) {
    utils::Vector source = utils::Vector<float>::Generate(M, nvals, seed).SortReduceDuplicates();
    source.Fill(utils::UniformRealGenerator<float>());

    auto spT = spla::Types::Float32(library);
    auto spV = spla::Vector::Make(M, spT, library);

    // Specify, that values already in row order + no duplicates
    auto spDesc = spla::Descriptor::Make(library);
    spDesc->SetParam(spla::Descriptor::Param::ValuesSorted);
    spDesc->SetParam(spla::Descriptor::Param::NoDuplicates);

    auto spDataSrc = spla::DataVector::Make(library);
    spDataSrc->SetRows(source.GetRows());
    spDataSrc->SetVals(source.GetVals());
    spDataSrc->SetNvals(source.GetNvals());

    auto spExprWrite = spla::Expression::Make(library);
    auto spWrite = spExprWrite->MakeDataWrite(spV, spDataSrc, spDesc);
    spExprWrite->Submit();
    spExprWrite->Wait();
    ASSERT_EQ(spExprWrite->GetState(), spla::Expression::State::Evaluated);

    utils::Vector<float> expected = source;
    ASSERT_TRUE(expected.Equals(spV));
}

void test(std::size_t M, std::size_t base, std::size_t step, std::size_t iter) {
    utils::testBlocks({100, 1000, 10000, 100000}, [=](spla::Library &library) {
        for (std::size_t i = 0; i < iter; i++) {
            std::size_t nvals = base + i * step;
            testCommon(library, M, nvals, i);
        }

        for (std::size_t i = 0; i < iter; i++) {
            std::size_t nvals = base + i * step;
            testSortedNoDuplicates(library, M, nvals, i);
        }
    });
}

TEST(DataVector, Small) {
    std::size_t M = 100;
    test(M, M, M, 10);
}

TEST(DataVector, Medium) {
    std::size_t M = 1030;
    test(M, M, M, 10);
}

TEST(DataVector, Large) {
    std::size_t M = 11300;
    test(M, M, M, 5);
}

SPLA_GTEST_MAIN