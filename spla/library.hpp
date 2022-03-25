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

#ifndef SPLA_LIBRARY_HPP
#define SPLA_LIBRARY_HPP

#include <iostream>
#include <memory>

#include <taskflow/taskflow.hpp>

#include <spla/config.hpp>
#include <spla/io/log.hpp>

#if defined(SPLA_BACKEND_REFERENCE)
    #include <spla/backend/reference/backend.hpp>
#elif defined(SPLA_BACKEND_OPENCL)
    #error "No backend for build"
#elif defined(SPLA_BACKEND_CUDA)
    #error "No backend for build"
#else
    #error "No backend for build"
#endif

namespace spla {

    /**
     * @addtogroup spla
     * @{
     */

    /**
     * @class Library
     * @brief Global library state
     *
     * Stores global library state and config.
     * Use this class to configure spla library for the usage.
     */
    class Library {
    public:
        explicit Library(const Config &config = Config())
            : m_config(config), m_block_factor(config.get_block_factor().value()) {

            // Default listener for std output
            log().add_listener([](const Log::Entry &entry) {
                std::stringstream output;
                output << "[" << to_string(entry.level) << "] "
                       << "[" << entry.file.substr(entry.file.find_last_of("/\\") + 1) << ":" << entry.line << "] "
                       << entry.message << "\n";

                if (entry.level == Log::Level::Error)
                    std::cerr << output.str();
                else
                    std::cout << output.str();
            });

            // Executor config
            assert(config.get_workers_count().has_value());
            m_executor = std::make_unique<tf::Executor>(config.get_workers_count().value());

            // Initialize backend
            backend::initialize(config);

            SPLA_LOG_INFO("init library backend: " << to_string(supported_backend()));
        }

        /** @return Storage split factor */
        [[nodiscard]] std::size_t block_factor() const { return m_block_factor; }

        /** @return Config used to init library */
        [[nodiscard]] const Config &config() const { return m_config; }

        /** @return Library executor for tasking */
        [[nodiscard]] tf::Executor &executor() const { return *m_executor; }

    private:
        Config m_config;
        std::size_t m_block_factor;
        std::unique_ptr<tf::Executor> m_executor;
    };

    /**
     * @brief Get global library ptr
     * @note For internal usage only
     *
     * @return Library ptr
     */
    inline std::unique_ptr<Library> &library_ptr() {
        static std::unique_ptr<Library> library;
        return library;
    }

    /**
     * @brief Get global library instance
     *
     * If library is not initialized, initializes new instance.
     * Use this method to access library and configure it before usage.
     *
     * @return Library instance
     */
    inline Library &library() {
        auto &library = library_ptr();

        if (!library)
            library = std::make_unique<Library>();

        return *library;
    }

    /**
     * @}
     */

}// namespace spla

#endif//SPLA_LIBRARY_HPP