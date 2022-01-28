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

#include <core/SplaError.hpp>
#include <core/SplaLibraryPrivate.hpp>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace {
    std::vector<boost::compute::device> FindAllDevices(const spla::Library::Config &config) {
        std::vector<boost::compute::device> devices;

        auto platforms = boost::compute::system::platforms();
        if (platforms.empty()) {
            RAISE_ERROR(DeviceNotPresent, "No OpenCL platform found");
        }

        if (!config.GetDeviceType().has_value() &&
            !config.GetPlatformName().has_value() &&
            config.GetDeviceAmount() == 1) {
            return {boost::compute::system::default_device()};
        }

        for (const boost::compute::platform &platform : platforms) {
            bool matchPlatform = !config.GetPlatformName().has_value() || platform.name().find(config.GetPlatformName().value()) != std::string::npos;

            if (!matchPlatform)
                continue;

            const auto &deviceType = config.GetDeviceType();

            for (const boost::compute::device &device : platform.devices()) {
                bool matchType = !deviceType.has_value() || ((deviceType.value() == spla::Library::Config::GPU && device.type() == boost::compute::device::type::gpu) ||
                                                             (deviceType.value() == spla::Library::Config::CPU && device.type() == boost::compute::device::type::cpu) ||
                                                             (deviceType.value() == spla::Library::Config::Accelerator && device.type() == boost::compute::device::type::accelerator));
                if (matchType) {
                    devices.push_back(device);
                }
            }
        }

        if (config.GetDeviceAmount().has_value() && devices.size() > config.GetDeviceAmount().value()) {
            devices.resize(config.GetDeviceAmount().value());
        }

        if (devices.empty()) {
            RAISE_ERROR(DeviceNotPresent, "No set of devices fits given constraints");
        }

        return devices;
    }

    boost::compute::platform GetDevicesPlatform(const std::vector<boost::compute::device> &devices) {
        if (devices.empty()) {
            RAISE_ERROR(DeviceNotPresent, "No OpenCL device has been found that meets these constraints");
        }
        auto platformId = devices[0].platform().id();
        for (const auto &device : devices) {
            if (platformId != device.platform().id()) {
                RAISE_ERROR(DeviceError, "Could not initialize context when devices has different platforms");
            }
        }
        return devices[0].platform();
    }

    std::shared_ptr<spdlog::logger> SetupLogger(const spla::Library::Config &config) {
        std::vector<spdlog::sink_ptr> sinks;

#ifdef SPLA_DEBUG
        auto consoleSinkLevel = spdlog::level::trace;
#else
        auto consoleSinkLevel = spdlog::level::err;
#endif

        auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        consoleSink->set_level(consoleSinkLevel);
        sinks.push_back(consoleSink);

        auto &filename = config.GetLogFilename();
        if (filename.has_value()) {
            auto fileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename.value());
            fileSink->set_level(spdlog::level::trace);
            sinks.push_back(fileSink);
        }

        auto logger = std::make_shared<spdlog::logger>("spla", sinks.begin(), sinks.end());
        logger->set_level(static_cast<spdlog::level::level_enum>(SPDLOG_ACTIVE_LEVEL));

        return logger;
    }
}// namespace

spla::LibraryPrivate::LibraryPrivate(
        spla::Library &library,
        spla::Library::Config config)
    : mDeviceManager(FindAllDevices(config)),
      mPlatform(GetDevicesPlatform(mDeviceManager.GetDevices())),
      mContext(mDeviceManager.GetDevices()),
      mContextConfig(std::move(config)) {
    mLogger = SetupLogger(mContextConfig);
    mDefaultDesc = Descriptor::Make(library);
    mExprManager = RefPtr<ExpressionManager>(new ExpressionManager(library));
    mAlgoManager = RefPtr<AlgorithmManager>(new AlgorithmManager(library));
    mExecutor = std::make_shared<tf::Executor>(mContextConfig.GetWorkersCount());

#if defined(SPLA_DEBUG) || defined(SPLA_DEBUG_RELEASE) || defined(SPLA_PROFILING) || defined(SPLA_PROFILING_TASKS)
    std::cout << "Platform: " << mPlatform.name() << " device count: " << mDeviceManager.GetDevices().size() << '\n';
#endif
}

spla::LibraryPrivate::~LibraryPrivate() {
    mExecutor.reset();
}

tf::Executor &spla::LibraryPrivate::GetTaskFlowExecutor() noexcept {
    return *mExecutor;
}

const spla::RefPtr<spla::Descriptor> &spla::LibraryPrivate::GetDefaultDesc() const noexcept {
    return mDefaultDesc;
}

const spla::RefPtr<spla::ExpressionManager> &spla::LibraryPrivate::GetExprManager() const noexcept {
    return mExprManager;
}

const spla::RefPtr<spla::AlgorithmManager> &spla::LibraryPrivate::GetAlgoManager() const noexcept {
    return mAlgoManager;
}

const std::vector<boost::compute::device> &spla::LibraryPrivate::GetDevices() const noexcept {
    return mDeviceManager.GetDevices();
}

spla::DeviceManager &spla::LibraryPrivate::GetDeviceManager() noexcept {
    return mDeviceManager;
}

const boost::compute::platform &spla::LibraryPrivate::GetPlatform() const noexcept {
    return mPlatform;
}

const boost::compute::context &spla::LibraryPrivate::GetContext() const noexcept {
    return mContext;
}

const spla::Library::Config &spla::LibraryPrivate::GetContextConfig() const noexcept {
    return mContextConfig;
}

const std::shared_ptr<spdlog::logger> &spla::LibraryPrivate::GetLogger() const noexcept {
    return mLogger;
}

std::unordered_map<std::string, spla::RefPtr<spla::Type>> &spla::LibraryPrivate::GetTypeCache() noexcept {
    return mTypeCache;
}

std::size_t spla::LibraryPrivate::GetBlockSize() const noexcept {
    return mContextConfig.GetBlockSize();
}