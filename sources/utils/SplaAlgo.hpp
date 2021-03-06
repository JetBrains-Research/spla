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

#ifndef SPLA_SPLAALGO_HPP
#define SPLA_SPLAALGO_HPP

#include <sstream>

#include <boost/hana.hpp>

#include <spla-cpp/SplaFunctionBinary.hpp>
#include <spla-cpp/SplaType.hpp>

namespace spla {

    /**
     * @addtogroup Internal
     * @{
     */

    namespace utils {

        /** @return Generic function `f x y = y` */
        inline RefPtr<FunctionBinary> MakeFunctionChooseSecond(const RefPtr<Type> &t) {
            std::stringstream body;

            body << "_ACCESS_B const uchar * b = (_ACCESS_B const uchar *)vp_b;\n"
                 << "_ACCESS_C uchar * c = (_ACCESS_C uchar *)vp_c;\n"
                 << "for (uint i = 0; i < " << t->GetByteSize() << "; i++)\n"
                 << "   c[i] = b[i];\n";

            return FunctionBinary::Make(t, t, t, body.str(), t->GetLibrary());
        }

    }// namespace utils

    /**
     * @}
     */

}// namespace spla

#endif//SPLA_SPLAALGO_HPP
