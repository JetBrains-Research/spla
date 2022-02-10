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

#ifndef SPLA_TESTING_HPP
#define SPLA_TESTING_HPP

#include <gtest/gtest.h>
#include <utils/Compute.hpp>
#include <utils/Matrix.hpp>
#include <utils/Operations.hpp>
#include <utils/Random.hpp>
#include <utils/Setup.hpp>
#include <utils/Typetraits.hpp>
#include <utils/Vector.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>

#define SPLA_TIME_BEGIN(timer) \
    auto timer = std::chrono::steady_clock::now();

#define SPLA_TIME_END(timer, message)                                           \
    std::cout << std::setw(12)                                                  \
              << message << " "                                                 \
              << std::setw(10)                                                  \
              << std::setprecision(9)                                           \
              << static_cast<double>(                                           \
                         std::chrono::duration_cast<std::chrono::microseconds>( \
                                 std::chrono::steady_clock::now() - timer)      \
                                 .count()) *                                    \
                         1e-3                                                   \
              << " ms" << std::endl;


// Put in the end of the unit test file
#define SPLA_GTEST_MAIN                                  \
    int main(int argc, char *argv[]) {                   \
        ::testing::GTEST_FLAG(catch_exceptions) = false; \
        ::testing::InitGoogleTest(&argc, argv);          \
        return RUN_ALL_TESTS();                          \
    }

#endif//SPLA_TESTING_HPP
