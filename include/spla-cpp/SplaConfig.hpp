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

#ifndef SPLA_SPLACONFIG_HPP
#define SPLA_SPLACONFIG_HPP

#include <string>

#ifdef SPLA_MSVC
    #ifdef SPLA_EXPORTS
        #define SPLA_API __declspec(dllexport)
    #else
        #define SPLA_API __declspec(dllimport)
    #endif
#else
    #define SPLA_API
#endif

#define SPLA_TEXT(text) u8##text

namespace spla {

    /**
     * @defgroup Internal
     *
     * @brief Implementation details
     *
     * @details The internal module implements the full functionality of the library.
     * It is not anticipated that the user will ever need to work with
     * the objects in this module, as it only contains details
     * of the library's implementation.
     */

#if defined(SPLA_TARGET_WINDOWS)
    /** String for universal file names representation */
    using Filename = std::wstring;
#elif defined(SPLA_TARGET_LINUX)
    /** String for universal file names representation */
    using Filename = std::string;
#elif defined(SPLA_TARGET_MACOSX)
    /** String for universal file names representation */
    using Filename = std::string;
#else
    #error Unsupported platfrom
#endif

    /** @brief Index of matrix/vector values */
    using Index = unsigned int;

    /** @brief Size (count) type */
    using Size = std::size_t;

    /**
     * @}
     */

}// namespace spla

#endif//SPLA_SPLACONFIG_HPP
