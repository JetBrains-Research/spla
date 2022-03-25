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

#ifndef SPLA_CONFIG_HPP
#define SPLA_CONFIG_HPP

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace spla {

    /**
     * @addtogroup spla
     * @{
     */

    /**
     * @class Backend
     * @brief Type of available backend for computations
     *
     * Library supports several distinct backends for computation.
     * Only single backend can be used for computations
     */
    enum class Backend {
        /** @brief Reference naive CPU backend for computations */
        Reference,
        /** @brief Cuda-based GPU backend */
        Cuda,
        /** @brief OpenCL-based GPU backend */
        OpenCL
    };

    /**
     * @brief Backend textual name
     * @param backend Backend type
     *
     * @return String representation
     */
    inline std::string to_string(Backend backend) {
        switch (backend) {
            case Backend::Reference:
                return "reference";
            case Backend::Cuda:
                return "cuda";
            case Backend::OpenCL:
                return "opencl";
            default:
                return "noop";
        }
    }

    /**
     * @brief Get supported backend for execution
     * @return Backend
     */
    inline Backend supported_backend() {
#if defined(SPLA_BACKEND_REFERENCE)
        return Backend::Reference;
#elif defined(SPLA_BACKEND_OPENCL)
        return Backend::OpenCL;
#elif defined(SPLA_BACKEND_CUDA)
        return Backend::Cuda;
#else
    #error "No backend for build"
#endif
    }

    /**
     * @class Config
     * @brief Config class used to configure library before usage
     */
    class Config {
    public:
        Config &set_workers_count(std::size_t workers) {
            assert(workers > 0);
            m_workers_count.emplace(workers);
            return *this;
        }

        Config &set_block_factor(std::size_t factor) {
            assert(factor > 0);
            m_block_factor.emplace(factor);
            return *this;
        }

        Config &set_opencl_vendor(std::string vendor) {
            m_opencl_vendor.emplace(std::move(vendor));
            return *this;
        }

        Config &set_opencl_devices(std::size_t devices) {
            assert(devices > 0);
            m_opencl_devices.emplace(devices);
            return *this;
        }

        Config &set_cuda_devices(std::size_t devices) {
            assert(devices > 0);
            m_cuda_devices.emplace(devices);
            return *this;
        }

        [[nodiscard]] const std::optional<std::size_t> &get_workers_count() const { return m_workers_count; }
        [[nodiscard]] const std::optional<std::size_t> &get_block_factor() const { return m_block_factor; }
        [[nodiscard]] const std::optional<std::string> &get_opencl_vendor() const { return m_opencl_vendor; }
        [[nodiscard]] const std::optional<std::size_t> &get_opencl_devices() const { return m_opencl_devices; }
        [[nodiscard]] const std::optional<std::size_t> &get_cuda_devices() const { return m_cuda_devices; }

    private:
        /* Common configuration */
        std::optional<std::size_t> m_workers_count{1};
        std::optional<std::size_t> m_block_factor{1};

        /* OpenCL related stuff */
        std::optional<std::string> m_opencl_vendor;
        std::optional<std::size_t> m_opencl_devices{1};

        /* Cuda stuff */
        std::optional<std::size_t> m_cuda_devices{1};
    };

    /**
     * @}
     */

}// namespace spla

#endif//SPLA_CONFIG_HPP