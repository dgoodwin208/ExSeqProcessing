#include "gtest/gtest.h"

#include "nearestinterp.h"
#include "cuda_task_executor.h"

#include "spdlog/spdlog.h"

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <random>

#include <thrust/host_vector.h>
#include <thrust/sequence.h>

namespace {

class NearestInterpTest : public ::testing::Test {
protected:
    std::shared_ptr<spdlog::logger> logger_;

    NearestInterpTest() {
        logger_ = spdlog::get("console");
        if (! logger_) {
            logger_ = spdlog::stdout_logger_mt("console");
        }
        //spdlog::set_level(spdlog::level::trace);
        spdlog::set_level(spdlog::level::info);
    }
    virtual ~NearestInterpTest() {
    }

    void check_results(
            cudautils::Sub2Ind& sub2ind,
            const unsigned int x_size,
            const unsigned int y_size,
            const unsigned int z_size,
            thrust::host_vector<int8_t>& map,
            thrust::host_vector<double>& img,
            thrust::host_vector<double>& interpolated_image) {

        for (int k = 0; k < z_size; k++) {
            for (int j = 0; j < y_size; j++) {
                for (int i = 0; i < x_size; i++) {
                    if (map[sub2ind(i, j, k)] > 0) {
                        ASSERT_EQ(img[sub2ind(i, j, k)], interpolated_image[sub2ind(i, j, k)]);
                    } else {
                        unsigned int x_start = max(i-1, 0);
                        unsigned int x_end   = min(i+1, x_size-1);
                        unsigned int y_start = max(j-1, 0);
                        unsigned int y_end   = min(j+1, y_size-1);
                        unsigned int z_start = max(k-1, 0);
                        unsigned int z_end   = min(k+1, z_size-1);
                        logger_->debug("x=({},{}) y=({},{}) z=({},{})", x_start, x_end, y_start, y_end, z_start, z_end);

                        double sum = 0.0;
                        unsigned int count = 0;
                        for (unsigned int w = z_start; w <= z_end; w++) {
                            for (unsigned int v = y_start; v <= y_end; v++) {
                                for (unsigned int u = x_start; u <= x_end; u++) {
                                    sum += img[sub2ind(u, v, w)] * double(map[sub2ind(u, v, w)]);
                                    count += (map[sub2ind(u, v, w)] > 0);
                                }
                            }
                        }

                        if (count > 0) {
                            double mean = sum / double(count);
                            ASSERT_EQ(mean, interpolated_image[sub2ind(i, j, k)]);
                            continue;
                        }

                        x_start = max(i-2, 0);
                        x_end   = min(i+2, x_size-1);
                        y_start = max(j-2, 0);
                        y_end   = min(j+2, y_size-1);
                        z_start = max(k-2, 0);
                        z_end   = min(k+2, z_size-1);
                        logger_->debug("x=({},{}) y=({},{}) z=({},{})", x_start, x_end, y_start, y_end, z_start, z_end);

                        sum = 0.0;
                        count = 0;
                        for (unsigned int w = z_start; w <= z_end; w++) {
                            for (unsigned int v = y_start; v <= y_end; v++) {
                                for (unsigned int u = x_start; u <= x_end; u++) {
                                    logger_->debug("({}, {}, {}); idx={}, map={}, img={}", u, v, w, sub2ind(u, v, w), map[sub2ind(u, v, w)], img[sub2ind(u, v, w)]);
                                    sum += img[sub2ind(u, v, w)] * double(map[sub2ind(u, v, w)]);
                                    count += (map[sub2ind(u, v, w)] > 0);
                                }
                            }
                        }

                        if (count > 0) {
                            double mean = sum / double(count);
                            ASSERT_EQ(mean, interpolated_image[sub2ind(i, j, k)]);
                        } else {
                            ASSERT_EQ(0.0, interpolated_image[sub2ind(i, j, k)]);
                        }
                    }
                }
            }
        }
    }
};


TEST_F(NearestInterpTest, Sub2Ind1Test) {

    cudautils::Sub2Ind sub2ind(3,4,5);

    ASSERT_EQ(0,  sub2ind(0,0,0));
    ASSERT_EQ(1,  sub2ind(1,0,0));
    ASSERT_EQ(7,  sub2ind(1,2,0));
    ASSERT_EQ(43, sub2ind(1,2,3));
}

TEST_F(NearestInterpTest, Sub2Ind2Test) {

    cudautils::Sub2Ind sub2ind(3,4,5,1,1,1);

    // 16 = sub2ind(1,1,1)
    ASSERT_EQ(16, sub2ind(0,0,0));
    ASSERT_EQ(17, sub2ind(1,0,0));
    ASSERT_EQ(23, sub2ind(1,2,0));
    ASSERT_EQ(59, sub2ind(1,2,3));
}

TEST_F(NearestInterpTest, Sub2Ind3Test) {

    cudautils::Sub2Ind sub2ind(3,4,5);
    sub2ind.setBasePos(1,1,1);

    // 9 = sub2ind(1,1,1)
    ASSERT_EQ(16, sub2ind(0,0,0));
    ASSERT_EQ(17, sub2ind(1,0,0));
    ASSERT_EQ(23, sub2ind(1,2,0));
    ASSERT_EQ(59, sub2ind(1,2,3));
}

TEST_F(NearestInterpTest, Ind2Sub1Test) {

    unsigned int x_stride = 4;
    unsigned int y_stride = 5;

    unsigned int x = 0;
    unsigned int y = 0;
    unsigned int z = 0;
    cudautils::ind2sub(x_stride, y_stride, 2, x, y, z);
    ASSERT_EQ(2, x);
    ASSERT_EQ(0, y);
    ASSERT_EQ(0, z);

    x = y = z = 0;
    cudautils::ind2sub(x_stride, y_stride, 7, x, y, z);
    ASSERT_EQ(3, x);
    ASSERT_EQ(1, y);
    ASSERT_EQ(0, z);

    x = y = z = 0;
    cudautils::ind2sub(x_stride, y_stride, 25, x, y, z);
    ASSERT_EQ(1, x);
    ASSERT_EQ(1, y);
    ASSERT_EQ(1, z);
}

TEST_F(NearestInterpTest, NearestInterpSmallImage_3x3_1Test) {

    const unsigned int x_size = 5;
    const unsigned int y_size = 5;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 5;
    const unsigned int y_sub_size = 5;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    img[sub2ind(2, 2, 2)] = 0.0;
    map[sub2ind(2, 2, 2)] = 0;

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpSmallImage_3x3_2Test) {

    const unsigned int x_size = 5;
    const unsigned int y_size = 5;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 5;
    const unsigned int y_sub_size = 5;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    img[sub2ind(0, 0, 0)] = 0.0;
    map[sub2ind(0, 0, 0)] = 0;

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpSmallImage_3x3_3Test) {

    const unsigned int x_size = 5;
    const unsigned int y_size = 5;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 5;
    const unsigned int y_sub_size = 5;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    img[sub2ind(4, 4, 4)] = 0.0;
    map[sub2ind(4, 4, 4)] = 0;

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpSmallImage_3x3_4Test) {

    const unsigned int x_size = 5;
    const unsigned int y_size = 5;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 5;
    const unsigned int y_sub_size = 5;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    img[sub2ind(0, 4, 4)] = 0.0;
    map[sub2ind(0, 4, 4)] = 0;

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpSmallImage_3x3_5Test) {

    const unsigned int x_size = 5;
    const unsigned int y_size = 5;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 5;
    const unsigned int y_sub_size = 5;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    img[sub2ind(4, 0, 4)] = 0.0;
    map[sub2ind(4, 0, 4)] = 0;

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpSmallImage_3x3_6Test) {

    const unsigned int x_size = 5;
    const unsigned int y_size = 5;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 5;
    const unsigned int y_sub_size = 5;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    img[sub2ind(4, 4, 0)] = 0.0;
    map[sub2ind(4, 4, 0)] = 0;

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpSmallImage_3x3_7Test) {

    const unsigned int x_size = 5;
    const unsigned int y_size = 5;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 5;
    const unsigned int y_sub_size = 5;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    img[sub2ind(0, 0, 4)] = 0.0;
    map[sub2ind(0, 0, 4)] = 0;

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpSmallImage_3x3_8Test) {

    const unsigned int x_size = 5;
    const unsigned int y_size = 5;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 5;
    const unsigned int y_sub_size = 5;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    img[sub2ind(0, 4, 0)] = 0.0;
    map[sub2ind(0, 4, 0)] = 0;

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpSmallImage_3x3_9Test) {

    const unsigned int x_size = 5;
    const unsigned int y_size = 5;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 5;
    const unsigned int y_sub_size = 5;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    img[sub2ind(4, 0, 0)] = 0.0;
    map[sub2ind(4, 0, 0)] = 0;

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpSmallImage_3x3_10Test) {

    const unsigned int x_size = 5;
    const unsigned int y_size = 5;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 5;
    const unsigned int y_sub_size = 5;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    img[sub2ind(1, 1, 1)] = 0.0;
    img[sub2ind(2, 1, 1)] = 0.0;
    map[sub2ind(1, 1, 1)] = 0;
    map[sub2ind(2, 1, 1)] = 0;

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpSmallImage_5x5_1Test) {

    const unsigned int x_size = 5;
    const unsigned int y_size = 5;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 5;
    const unsigned int y_sub_size = 5;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    for (unsigned int k = 0; k < 3; k++) {
        for (unsigned int j = 0; j < 3; j++) {
            for (unsigned int i = 0; i < 3; i++) {
                img[sub2ind(i, j, k)] = 0.0;
                map[sub2ind(i, j, k)] = 0;
            }
        }
    }

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpSmallImage_5x5_2Test) {

    const unsigned int x_size = 5;
    const unsigned int y_size = 5;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 5;
    const unsigned int y_sub_size = 5;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    for (unsigned int k = 2; k < 5; k++) {
        for (unsigned int j = 2; j < 5; j++) {
            for (unsigned int i = 2; i < 5; i++) {
                img[sub2ind(i, j, k)] = 0.0;
                map[sub2ind(i, j, k)] = 0;
            }
        }
    }

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpSmallImage_5x5_3Test) {

    const unsigned int x_size = 5;
    const unsigned int y_size = 5;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 5;
    const unsigned int y_sub_size = 5;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    for (unsigned int k = 0; k < 3; k++) {
        for (unsigned int j = 2; j < 5; j++) {
            for (unsigned int i = 2; i < 5; i++) {
                img[sub2ind(i, j, k)] = 0.0;
                map[sub2ind(i, j, k)] = 0;
            }
        }
    }

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpSmallImage_5x5_4Test) {

    const unsigned int x_size = 5;
    const unsigned int y_size = 5;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 5;
    const unsigned int y_sub_size = 5;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    for (unsigned int k = 2; k < 5; k++) {
        for (unsigned int j = 0; j < 3; j++) {
            for (unsigned int i = 2; i < 5; i++) {
                img[sub2ind(i, j, k)] = 0.0;
                map[sub2ind(i, j, k)] = 0;
            }
        }
    }

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpSmallImage_5x5_5Test) {

    const unsigned int x_size = 5;
    const unsigned int y_size = 5;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 5;
    const unsigned int y_sub_size = 5;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    for (unsigned int k = 2; k < 5; k++) {
        for (unsigned int j = 2; j < 5; j++) {
            for (unsigned int i = 0; i < 3; i++) {
                img[sub2ind(i, j, k)] = 0.0;
                map[sub2ind(i, j, k)] = 0;
            }
        }
    }

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpSmallImage_5x5_6Test) {

    const unsigned int x_size = 5;
    const unsigned int y_size = 5;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 5;
    const unsigned int y_sub_size = 5;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    for (unsigned int k = 0; k < 3; k++) {
        for (unsigned int j = 0; j < 3; j++) {
            for (unsigned int i = 2; i < 5; i++) {
                img[sub2ind(i, j, k)] = 0.0;
                map[sub2ind(i, j, k)] = 0;
            }
        }
    }

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpSmallImage_5x5_7Test) {

    const unsigned int x_size = 5;
    const unsigned int y_size = 5;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 5;
    const unsigned int y_sub_size = 5;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    for (unsigned int k = 0; k < 3; k++) {
        for (unsigned int j = 2; j < 5; j++) {
            for (unsigned int i = 0; i < 3; i++) {
                img[sub2ind(i, j, k)] = 0.0;
                map[sub2ind(i, j, k)] = 0;
            }
        }
    }

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpSmallImage_5x5_8Test) {

    const unsigned int x_size = 5;
    const unsigned int y_size = 5;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 5;
    const unsigned int y_sub_size = 5;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    for (unsigned int k = 2; k < 5; k++) {
        for (unsigned int j = 0; j < 3; j++) {
            for (unsigned int i = 0; i < 3; i++) {
                img[sub2ind(i, j, k)] = 0.0;
                map[sub2ind(i, j, k)] = 0;
            }
        }
    }

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpMiddleImage_1Test) {

    const unsigned int x_size = 10;
    const unsigned int y_size = 5;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 10;
    const unsigned int y_sub_size = 5;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    for (unsigned int k = 0; k < 3; k++) {
        for (unsigned int j = 0; j < 3; j++) {
            for (unsigned int i = 0; i < 3; i++) {
                img[sub2ind(i, j, k)] = 0.0;
                map[sub2ind(i, j, k)] = 0;
            }
        }
    }

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpMiddleImage_2Test) {

    const unsigned int x_size = 10;
    const unsigned int y_size = 5;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 10;
    const unsigned int y_sub_size = 5;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    for (unsigned int k = 0; k < 3; k++) {
        for (unsigned int j = 0; j < 3; j++) {
            for (unsigned int i = 4; i < 7; i++) {
                img[sub2ind(i, j, k)] = 0.0;
                map[sub2ind(i, j, k)] = 0;
            }
        }
    }

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpMiddleImage_3Test) {

    const unsigned int x_size = 5;
    const unsigned int y_size = 10;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 5;
    const unsigned int y_sub_size = 10;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    for (unsigned int k = 0; k < 3; k++) {
        for (unsigned int j = 0; j < 3; j++) {
            for (unsigned int i = 0; i < 3; i++) {
                img[sub2ind(i, j, k)] = 0.0;
                map[sub2ind(i, j, k)] = 0;
            }
        }
    }

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpMiddleImage_4Test) {

    const unsigned int x_size = 5;
    const unsigned int y_size = 10;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 5;
    const unsigned int y_sub_size = 10;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    for (unsigned int k = 0; k < 3; k++) {
        for (unsigned int j = 4; j < 7; j++) {
            for (unsigned int i = 0; i < 3; i++) {
                img[sub2ind(i, j, k)] = 0.0;
                map[sub2ind(i, j, k)] = 0;
            }
        }
    }

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpMiddleImage_5Test) {

    const unsigned int x_size = 10;
    const unsigned int y_size = 10;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 10;
    const unsigned int y_sub_size = 10;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    for (unsigned int k = 0; k < 3; k++) {
        for (unsigned int j = 4; j < 7; j++) {
            for (unsigned int i = 4; i < 7; i++) {
                img[sub2ind(i, j, k)] = 0.0;
                map[sub2ind(i, j, k)] = 0;
            }
        }
    }

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpMiddleImage_6Test) {

    const unsigned int x_size = 20;
    const unsigned int y_size = 20;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 10;
    const unsigned int y_sub_size = 10;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    for (unsigned int k = 0; k < 3; k++) {
        for (unsigned int j = 4; j < 7; j++) {
            for (unsigned int i = 4; i < 7; i++) {
                img[sub2ind(i, j, k)] = 0.0;
                map[sub2ind(i, j, k)] = 0;
            }
        }
    }

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpMiddleImage_7Test) {

    const unsigned int x_size = 20;
    const unsigned int y_size = 20;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 10;
    const unsigned int y_sub_size = 10;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 2;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    for (unsigned int k = 0; k < 3; k++) {
        for (unsigned int j = 4; j < 7; j++) {
            for (unsigned int i = 4; i < 7; i++) {
                img[sub2ind(i, j, k)] = 0.0;
                map[sub2ind(i, j, k)] = 0;
            }
        }
    }

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

TEST_F(NearestInterpTest, NearestInterpMiddleImage_8Test) {

    const unsigned int x_size = 20;
    const unsigned int y_size = 20;
    const unsigned int z_size = 5;
    const unsigned int x_sub_size = 10;
    const unsigned int y_sub_size = 10;
    const unsigned int dx = 5;
    const unsigned int dy = 5;
    const unsigned int dw = 2;

    const unsigned int num_gpus = 2;
    const unsigned int num_streams = 2;

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::sequence(img.begin(), img.end());

    cudautils::Sub2Ind sub2ind(x_size, y_size, z_size);
    for (unsigned int k = 0; k < 3; k++) {
        for (unsigned int j = 4; j < 7; j++) {
            for (unsigned int i = 4; i < 7; i++) {
                img[sub2ind(i, j, k)] = 0.0;
                map[sub2ind(i, j, k)] = 0;
            }
        }
    }

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);
    ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));

    check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);
}

}

