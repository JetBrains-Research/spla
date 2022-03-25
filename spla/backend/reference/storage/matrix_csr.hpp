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

#ifndef SPLA_REFERENCE_MATRIX_CSR_HPP
#define SPLA_REFERENCE_MATRIX_CSR_HPP

#include <cassert>
#include <utility>
#include <vector>

#include <spla/detail/matrix_block.hpp>

namespace spla::backend {

    /**
     * @addtogroup reference
     * @{
     */

    template<typename T>
    class MatrixCsr final : public detail::MatrixBlock<T> {
    public:
        MatrixCsr(std::size_t nrows, std::size_t ncols, std::size_t nvals,
                  std::vector<Index> rows_ptr,
                  std::vector<Index> cols_index,
                  std::vector<T> values)
            : detail::MatrixBlock<T>(nrows, ncols, nvals),
              m_rows_ptr(std::move(rows_ptr)),
              m_cols_index(std::move(cols_index)),
              m_values(values) {
            assert(m_rows_ptr.size() == nrows + 1);
            assert(m_cols_index.size() == nvals);
            assert(m_values.size() == nvals || !type_has_values<T>());
        }

        ~MatrixCsr() override = default;

        [[nodiscard]] const std::vector<Index> &rows_ptr() const { return m_rows_ptr; }
        [[nodiscard]] const std::vector<Index> &cols_index() const { return m_cols_index; }
        [[nodiscard]] const std::vector<T> &values() const { return m_values; }

    private:
        std::vector<Index> m_rows_ptr;
        std::vector<Index> m_cols_index;
        std::vector<T> m_values;
    };

    /**
     * @}
     */

}// namespace spla::backend

#endif//SPLA_REFERENCE_MATRIX_CSR_HPP
