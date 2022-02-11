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

#include <boost/compute.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

TEST(Basic, BoostExample) {
    namespace compute = boost::compute;

    // get the default compute device
    compute::device gpu = compute::system::default_device();

    // create a compute context and command queue
    compute::context ctx(gpu);
    compute::command_queue queue(ctx, gpu);

    std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
    auto dist = std::uniform_real_distribution<float>(0.0, 100.0);

    // generate random numbers on the host
    std::vector<float> host_vector(1000000);
    std::generate(host_vector.begin(), host_vector.end(), [&]() { return dist(engine); });

    // create vector on the device
    compute::vector<float> device_vector(1000000, ctx);

    // copy data to the device
    compute::copy(host_vector.begin(), host_vector.end(), device_vector.begin(), queue);

    // sort data on the device
    compute::sort(device_vector.begin(), device_vector.end(), queue);

    // copy data back to the host
    compute::copy(device_vector.begin(), device_vector.end(), host_vector.begin(), queue);

    queue.finish();

    // Ensure, that data is sorted
    float prev = -1;
    for (auto v : host_vector) {
        EXPECT_LE(prev, v);
        prev = v;
    }
}

TEST(Basic, DISABLED_PriceCost) {
    namespace compute = boost::compute;

    // get default device and setup context
    compute::device gpu = compute::system::default_device();
    compute::context context(gpu);
    compute::command_queue queue(context, gpu);

    // prices #1 (from 10.0 to 11.0)
    std::vector<float> prices1;
    for (float i = 10.0; i <= 11.0; i += 0.1) {
        prices1.push_back(i);
    }

    // prices #2 (from 11.0 to 10.0)
    std::vector<float> prices2;
    for (float i = 11.0; i >= 10.0; i -= 0.1) {
        prices2.push_back(i);
    }

    // create gpu vectors
    compute::vector<float> gpu_prices1(prices1.size(), context);
    compute::vector<float> gpu_prices2(prices2.size(), context);

    // copy prices to gpu
    compute::copy(prices1.begin(), prices1.end(), gpu_prices1.begin(), queue);
    compute::copy(prices2.begin(), prices2.end(), gpu_prices2.begin(), queue);

    // function returning true if the second price is less than the first price
    BOOST_COMPUTE_FUNCTION(bool, check_price_cross, (boost::tuple<float, float> prices),
                           {
                               // first price
                               const float first = boost_tuple_get(prices, 0);

                               // second price
                               const float second = boost_tuple_get(prices, 1);

                               // return true if second price is less than first
                               return second < first;
                           });

    // find cross point (should be 10.5)
    compute::vector<float>::iterator iter = boost::get<0>(
            compute::find_if(
                    compute::make_zip_iterator(
                            boost::make_tuple(gpu_prices1.begin(), gpu_prices2.begin())),
                    compute::make_zip_iterator(
                            boost::make_tuple(gpu_prices1.end(), gpu_prices2.end())),
                    check_price_cross,
                    queue)
                    .get_iterator_tuple());

    // print out result
    int index = std::distance(gpu_prices1.begin(), iter);
    std::cout << "price cross at index: " << index << std::endl;

    float value;
    compute::copy_n(iter, 1, &value, queue);
    std::cout << "value: " << value << std::endl;
}

TEST(Basic, SortIndices) {
    namespace compute = boost::compute;

    compute::device gpu = compute::system::default_device();
    compute::context ctx(gpu);
    compute::command_queue queue(ctx, gpu);

    std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
    auto indicesCount = 100;
    auto dist = std::uniform_int_distribution<int>(0, indicesCount);
    auto generator = [&]() { return dist(engine); };

    std::vector<int> I(indicesCount);
    std::vector<int> J(indicesCount);

    std::generate(I.begin(), I.end(), generator);
    std::generate(J.begin(), J.end(), generator);

    compute::vector<int> deviceI(indicesCount, ctx);
    compute::vector<int> deviceJ(indicesCount, ctx);

    compute::copy(I.begin(), I.end(), deviceI.begin(), queue);
    compute::copy(J.begin(), J.end(), deviceJ.begin(), queue);

    compute::vector<int> permutation(indicesCount, ctx);
    compute::copy(compute::counting_iterator<int>(0), compute::counting_iterator<int>(indicesCount), permutation.begin(), queue);

    compute::vector<int> temp(indicesCount, ctx);
    compute::copy(deviceJ.begin(), deviceJ.end(), temp.begin(), queue);

    compute::sort_by_key(temp.begin(), temp.end(), permutation.begin(), queue);
    compute::copy(deviceI.begin(), deviceI.end(), temp.begin(), queue);
    compute::gather(permutation.begin(), permutation.end(), temp.begin(), deviceI.begin(), queue);

    compute::sort_by_key(deviceI.begin(), deviceI.end(), permutation.begin(), queue);
    compute::copy(deviceJ.begin(), deviceJ.end(), temp.begin(), queue);
    compute::gather(permutation.begin(), permutation.end(), temp.begin(), deviceJ.begin(), queue);

    compute::copy(deviceI.begin(), deviceI.end(), I.begin(), queue);
    compute::copy(deviceJ.begin(), deviceJ.end(), J.begin(), queue);

    queue.finish();

    int i = -1, j = -1;
    for (auto k = 0; k < indicesCount; k++) {
        EXPECT_TRUE(i <= I[k] || (i == I[k] && j <= J[k]));
        i = I[k];
        j = J[k];
    }
}

TEST(Basic, TrnasformMGPU) {
    namespace compute = boost::compute;

    std::size_t N = 1000000000;
    std::size_t N2 = N / 2;
    std::size_t R = 10;
    auto platform = compute::system::default_device().platform();
    auto devices = platform.devices();
    auto ctx = compute::context(devices);

    std::cout << "default platform: " << platform.name() << std::endl;
    std::cout << "devices: " << devices.size() << std::endl;

    using compute::lambda::_1;

    if (devices.size() == 2) {
        compute::command_queue q1(ctx, devices[0]);
        compute::command_queue q2(ctx, devices[1]);

        std::vector<unsigned int> source(N);
        for (std::size_t i = 0; i < N; i++)
            source[i] = (i ^ 0xf0f0abeu) + 13 * i;

        {
            compute::vector<unsigned int> buff(N, ctx);
            compute::vector<unsigned int> res(N, ctx);
            compute::copy(source.begin(), source.end(), buff.begin(), q1);
            for (std::size_t i = 0; i < R; i++) {
                SPLA_TIME_BEGIN(t);
                compute::transform(buff.begin(), buff.end(), res.begin(), _1 + _1, q1);
                q1.finish();
                SPLA_TIME_END(t, "d1");
            }
        }

        {
            compute::vector<unsigned int> buff(N, ctx);
            compute::vector<unsigned int> res(N2, ctx);
            compute::copy(source.begin(), source.end(), buff.begin(), q1);
            for (std::size_t i = 0; i < R; i++) {
                SPLA_TIME_BEGIN(t);
                compute::transform(buff.begin(), buff.begin() + N2, res.begin(), _1 + _1, q1);
                q1.finish();
                SPLA_TIME_END(t, "d1-h");
            }
        }

        {
            compute::vector<unsigned int> buff(N, ctx);
            compute::vector<unsigned int> res1(N2, ctx);
            compute::vector<unsigned int> res2(N2, ctx);
            compute::copy(source.begin(), source.end(), buff.begin(), q1);
            for (std::size_t i = 0; i < R; i++) {
                SPLA_TIME_BEGIN(t);
                compute::transform(buff.begin() + 0 * N2, buff.begin() + 1 * N2, res1.begin(), _1 + _1, q1);
                compute::transform(buff.begin() + 1 * N2, buff.begin() + 2 * N2, res2.begin(), _1 + _1, q2);
                q1.finish();
                q2.finish();
                SPLA_TIME_END(t, "d2-r1");
            }
        }

        {
            compute::vector<unsigned int> buff1(N2, ctx);
            compute::vector<unsigned int> buff2(N2, ctx);
            compute::vector<unsigned int> res1(N2, ctx);
            compute::vector<unsigned int> res2(N2, ctx);
            compute::copy(source.begin() + 0 * N2, source.begin() + 1 * N2, buff1.begin(), q1);
            compute::copy(source.begin() + 1 * N2, source.begin() + 2 * N2, buff2.begin(), q2);
            for (std::size_t i = 0; i < R; i++) {
                SPLA_TIME_BEGIN(t);
                compute::transform(buff1.begin(), buff1.end(), res1.begin(), _1 + _1, q1);
                compute::transform(buff2.begin(), buff2.end(), res2.begin(), _1 + _1, q2);
                q1.finish();
                q2.finish();
                SPLA_TIME_END(t, "d2-r2");
            }
        }

        {
            compute::vector<unsigned int> buff1(N2, ctx);
            compute::vector<unsigned int> buff2(N2, ctx);
            compute::vector<unsigned int> res1(N2, ctx);
            compute::vector<unsigned int> res2(N2, ctx);
            compute::copy(source.begin() + 0 * N2, source.begin() + 1 * N2, buff1.begin(), q1);
            compute::copy(source.begin() + 1 * N2, source.begin() + 2 * N2, buff2.begin(), q2);

            std::atomic_bool f1{false};
            std::atomic_bool f2{false};
            std::atomic_bool s1{false};
            std::atomic_bool s2{false};
            std::thread t1([&]() {
                while (!f1.load()) {
                    if (s1.load()) {
                        compute::transform(buff1.begin(), buff1.end(), res1.begin(), _1 + _1, q1);
                        q1.finish();
                        s1.store(false);
                    }
                }
            });
            std::thread t2([&]() {
                while (!f2.load()) {
                    if (s2.load()) {
                        compute::transform(buff2.begin(), buff2.end(), res2.begin(), _1 + _1, q2);
                        q2.finish();
                        s2.store(false);
                    }
                }
            });

            for (std::size_t i = 0; i < R; i++) {
                SPLA_TIME_BEGIN(t);
                s1.store(true);
                s2.store(true);
                while (s1.load() || s2.load()) {}
                SPLA_TIME_END(t, "d2-r2-mt");
            }
            f1.store(true);
            f2.store(true);
            t1.join();
            t2.join();
        }
    }
}

SPLA_GTEST_MAIN