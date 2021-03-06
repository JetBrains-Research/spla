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

#include <cxxopts.hpp>

#include <spla-algo/SplaAlgo.hpp>
#include <spla-cpp/Spla.hpp>

int main(int argc, const char *const *argv) {
    cxxopts::Options options("spla-tc", "TC (triangles counting) algorithm with spla library");
    options.add_option("", cxxopts::Option("h,help", "display help info", cxxopts::value<bool>()->default_value("false")));
    options.add_option("", cxxopts::Option("mtxpath", "path to matrix file", cxxopts::value<std::string>()));
    options.add_option("", cxxopts::Option("niters", "number of iterations to run", cxxopts::value<int>()->default_value("4")));
    options.add_option("", cxxopts::Option("bsize", "size of block to store matrix/vector", cxxopts::value<int>()->default_value("10000000")));
    options.add_option("", cxxopts::Option("devices-count", "amount of devices for execution", cxxopts::value<int>()->default_value("1")));
    options.add_option("", cxxopts::Option("verbose", "verbose std output", cxxopts::value<bool>()->default_value("true")));
    options.add_option("", cxxopts::Option("platform", "opencl platform to select", cxxopts::value<std::string>()->default_value("")));
    options.add_option("", cxxopts::Option("debug-timing", "timing for each iteration of algorithm", cxxopts::value<bool>()->default_value("false")));
    cxxopts::ParseResult args;

    try {
        args = options.parse(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "Failed parse input arguments: " << e.what();
        return 1;
    }

    if (args["help"].as<bool>()) {
        std::cout << options.help();
        return 0;
    }

    std::string mtxpath;
    int niters;
    int bsize;
    int devicesCount;
    bool removeLoops = true;
    bool ignoreValues = true;
    bool verbose;
    bool debugTiming;
    std::string platform;

    try {
        mtxpath = args["mtxpath"].as<std::string>();
        niters = args["niters"].as<int>();
        bsize = args["bsize"].as<int>();
        devicesCount = args["devices-count"].as<int>();
        verbose = args["verbose"].as<bool>();
        debugTiming = args["debug-timing"].as<bool>();
        platform = args["platform"].as<std::string>();
    } catch (const std::exception &e) {
        std::cerr << "Invalid input arguments: " << e.what();
        return 1;
    }

    assert(niters > 0);
    assert(bsize > 1);
    assert(devicesCount >= 1);

    // Load data
    spla::MatrixLoader<std::int32_t> loader;

    try {
        loader.Load(mtxpath, true, removeLoops, ignoreValues, verbose);
    } catch (const std::exception &e) {
        std::cerr << "Failed load matrix: " << e.what();
        return 1;
    }

    assert(loader.GetNrows() == loader.GetNcols());

    // Fill matrix uniformly with 1
    loader.Fill(1);

    // Configure library
    spla::Library::Config config;
    config.SetDeviceType(spla::Library::Config::DeviceType::GPU);
    config.SetBlockSize(bsize);
    config.SetWorkersCount(devicesCount * devicesCount);
    config.LimitAmount(devicesCount);
    config.SetPlatform(platform);
    spla::Library library(config);

    // Output info about devices for evaluation
    if (verbose)
        std::cout << library.PrintContextConfig() << std::endl;

    // A and B tc args
    auto M = loader.GetNrows(), N = loader.GetNcols();
    auto T = spla::Types::Int32(library);
    spla::RefPtr<spla::Matrix> A = spla::Matrix::Make(M, N, T, library);
    spla::RefPtr<spla::Matrix> L = spla::Matrix::Make(M, N, T, library);
    spla::RefPtr<spla::Matrix> U = spla::Matrix::Make(M, N, T, library);

    // Data is sorted without duplicated values
    spla::RefPtr<spla::Descriptor> dataDesc = spla::Descriptor::Make(library);
    dataDesc->SetParam(spla::Descriptor::Param::ValuesSorted);
    dataDesc->SetParam(spla::Descriptor::Param::NoDuplicates);

    // Prepare data, fill A, and get L and U matrices
    spla::RefPtr<spla::Expression> prepareData = spla::Expression::Make(library);
    auto writeA = prepareData->MakeDataWrite(A, spla::DataMatrix::Make(loader.GetRowIndices().data(), loader.GetColIndices().data(), loader.GetValues().data(), loader.GetNvals(), library), dataDesc);
    prepareData->Dependency(writeA, prepareData->MakeTril(L, A));
    prepareData->Dependency(writeA, prepareData->MakeTriu(U, A));
    prepareData->SubmitWait();
    SPLA_ALGO_CHECK(prepareData);

    // Release A to free unused in algo memory
    A.Reset();

    // Set for U transposed decoration U^T (in case of undirected graph exactly L)
    U->SetDecoration(spla::Decorated::Decoration::TransposedMatrix, L.As<spla::Object>());

    // Descriptor to profile time if required
    spla::RefPtr<spla::Descriptor> desc = spla::Descriptor::Make(library);
    desc->SetParam(spla::Descriptor::Param::ProfileTime, debugTiming);

    std::int32_t nTrins = 0;

    // Warm up phase
    spla::CpuTimer tWarmUp;
    {
        spla::RefPtr<spla::Matrix> B = spla::Matrix::Make(M, N, T, library);
        tWarmUp.Start();
        spla::Tc(nTrins, B, L, U, desc);
        tWarmUp.Stop();
    }

    // Main phase, measure iterations
    std::vector<spla::CpuTimer> tIters(niters);
    for (int i = 0; i < niters; i++) {
        spla::CpuTimer &tIter = tIters[i];
        {
            spla::RefPtr<spla::Matrix> B = spla::Matrix::Make(M, N, T, library);
            tIter.Start();
            spla::Tc(nTrins, B, L, U, desc);
            tIter.Stop();
        }
    }

    spla::OutputMeasurements(tWarmUp, tIters);
    return 0;
}
