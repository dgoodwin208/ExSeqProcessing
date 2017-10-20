#include "gtest/gtest.h"

#include "gpudevice.h"
#include "radixsort.h"

#include "spdlog/spdlog.h"

#include <vector>
#include <cstdint>
#include <random>

namespace {

class GpuDeviceTest : public ::testing::Test {
protected:
    GpuDeviceTest() {
        auto logger = spdlog::basic_logger_mt("mex_logger", "mex.log");
    }
    virtual ~GpuDeviceTest() {
    }
};

TEST_F(GpuDeviceTest, GetGpuNumTest) {
    int gpu_num = cudautils::get_gpu_num();

    EXPECT_EQ(2, gpu_num);
}

class RadixSortTest : public ::testing::Test {
protected:
    RadixSortTest() {
    }
    virtual ~RadixSortTest() {
    }
};

TEST_F(RadixSortTest, GpuSortAndResortTest) {
    std::vector<uint16_t> keys;
    std::vector<unsigned int> values;

    const size_t DATA_SIZE = 10000;

    std::mt19937 mt(1);
    for (size_t i = 0; i < DATA_SIZE; i++) {
        keys.push_back(mt());
        values.push_back(i);
    }
    std::vector<uint16_t> keys2(keys.begin(), keys.end());

    cudautils::radixsort(keys2, values);

    for (size_t i = 0, j = 1; i < DATA_SIZE - 1; i++, j++) {
        ASSERT_LE(keys2[i], keys2[j]);
    }

    std::vector<double> keys3;
    for (size_t i = 0; i < DATA_SIZE; i++) {
        keys3.push_back((double)keys2[i]);
    }

    cudautils::radixsort(values, keys3);

    for (size_t i = 0, j = 1; i < DATA_SIZE - 1; i++, j++) {
        ASSERT_LE(values[i], values[j]);
    }
    for (size_t i = 0; i < DATA_SIZE; i++) {
        ASSERT_EQ((double)keys[i], keys3[i]);
    }
}

TEST_F(RadixSortTest, HostSortAndResortTest) {
    std::vector<uint16_t> keys;
    std::vector<unsigned int> values;

    const size_t DATA_SIZE = 10000;

    std::mt19937 mt(1);
    for (size_t i = 0; i < DATA_SIZE; i++) {
        keys.push_back(mt());
        values.push_back(i);
    }
    std::vector<uint16_t> keys2(keys.begin(), keys.end());

    cudautils::radixsort_host(keys2, values);

    for (size_t i = 0, j = 1; i < DATA_SIZE - 1; i++, j++) {
        ASSERT_LE(keys2[i], keys2[j]);
    }

    std::vector<double> keys3;
    for (size_t i = 0; i < DATA_SIZE; i++) {
        keys3.push_back((double)keys2[i]);
    }

    cudautils::radixsort_host(values, keys3);

    for (size_t i = 0, j = 1; i < DATA_SIZE - 1; i++, j++) {
        ASSERT_LE(values[i], values[j]);
    }
    for (size_t i = 0; i < DATA_SIZE; i++) {
        ASSERT_EQ((double)keys[i], keys3[i]);
    }
}

TEST_F(RadixSortTest, GpuOutOfMemoryTest) {
    std::vector<unsigned int> keys;
    std::vector<double> values;

    const size_t DATA_SIZE = 1024*1024*200;

    std::mt19937 mt(1);
    for (size_t i = 0; i < DATA_SIZE; i++) {
        keys.push_back(mt());
        values.push_back((double)i);
    }

    ASSERT_NO_THROW(cudautils::radixsort(keys, values));

    keys.resize(DATA_SIZE * 2);
    values.resize(DATA_SIZE * 2);

    for (size_t i = 0; i < DATA_SIZE; i++) {
        keys.push_back(mt());
        values.push_back((double)(i + DATA_SIZE));
    }

    EXPECT_THROW(cudautils::radixsort(keys, values), std::bad_alloc);
}

}


