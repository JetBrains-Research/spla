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

#ifndef SPLA_READ_HPP
#define SPLA_READ_HPP

#include <algorithm>
#include <vector>

#include <spla/backend.hpp>
#include <spla/descriptor.hpp>
#include <spla/expression_node.hpp>
#include <spla/types.hpp>
#include <spla/vector.hpp>

#include <spla/detail/storage_utils.hpp>

namespace spla::expression {

    /**
     * @addtogroup internal
     * @{
     */

    template<typename T, typename Callback>
    class ReadVector final : public ExpressionNode {
    public:
        ReadVector(Vector<T> vector, Callback callback, const Descriptor &desc, std::size_t id, Expression &expression)
            : ExpressionNode(desc, id, expression), m_vector(std::move(vector)), m_callback(std::move(callback)) {}

        ~ReadVector() override = default;

        std::string type() const override {
            return "read-vector";
        }

    private:
        void prepare() override {
            m_vector.storage()->lock_read();
        }

        void finalize() override {
            m_vector.storage()->unlock_read();
        }

        void execute(detail::SubtaskBuilder &builder) override {
            auto storage = m_vector.storage();
            auto nvals = storage->nvals();
            auto host_rows = std::make_shared<std::vector<Index>>(nvals);
            auto host_values = std::make_shared<std::vector<T>>(type_has_values<T>() ? nvals : 0);
            auto offsets = detail::storage_block_offsets(storage->blocks());

            auto notify = builder.emplace("read-notify", [this, host_rows, host_values]() {
                m_callback(*host_rows, *host_values);
            });

            for (std::size_t i = 0; i < storage->block_count_rows(); i++) {
                builder.emplace("read-block", [storage, host_rows, host_values, offsets, i, this]() {
                           auto block = storage->block(i);

                           if (block.is_null())
                               return;

                           auto blockSize = storage->block_size();
                           auto schema = storage->schema();
                           backend::ReadParams readParams{detail::block_offset(blockSize, i), offsets[i]};
                           backend::DispatchParams dispatchParams{i, i};

                           if (schema == VectorSchema::Sparse)
                               backend::read(block.template cast<backend::VectorCoo<T>>(), *host_rows, *host_values, desc(), readParams, dispatchParams);
                           if (schema == VectorSchema::Dense)
                               backend::read(block.template cast<backend::VectorDense<T>>(), *host_rows, *host_values, desc(), readParams, dispatchParams);
                       })
                        .precede(notify);
            }
        }

    private:
        Vector<T> m_vector;
        Callback m_callback;
    };

    /**
     * @}
     */

}// namespace spla::expression

#endif//SPLA_READ_HPP