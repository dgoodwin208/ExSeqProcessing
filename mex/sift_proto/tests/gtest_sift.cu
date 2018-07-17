#include "gtest/gtest.h"

#include "sift.h"
#include "cuda_task_executor.h"
#include <vector>
#include <iterator>
#include <algorithm>
#include <cublas_v2.h>

#include "spdlog/spdlog.h"

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <random>

#include <thrust/host_vector.h>

namespace {

class SiftTest : public ::testing::Test {
protected:
    std::shared_ptr<spdlog::logger> logger_;

    SiftTest() {
        logger_ = spdlog::get("console");
        if (! logger_) {
            logger_ = spdlog::stdout_logger_mt("console");
        }
        //spdlog::set_level(spdlog::level::trace);
        spdlog::set_level(spdlog::level::info);
    }
    virtual ~SiftTest() {
    }

    template <class T>
    void print_arr(T * arr, int N) {
        for (int i=0; i < N; i++) {
            logger_->debug("\t[{}]={}", i, arr[i]);
        }
    }

};


TEST_F(SiftTest, Sub2Ind1Test) {

    cudautils::Sub2Ind sub2ind(3,4,5);

    ASSERT_EQ(0,  sub2ind(0,0,0));
    ASSERT_EQ(1,  sub2ind(1,0,0));
    ASSERT_EQ(7,  sub2ind(1,2,0));
    ASSERT_EQ(43, sub2ind(1,2,3));
}

TEST_F(SiftTest, Sub2Ind2Test) {

    cudautils::Sub2Ind sub2ind(3,4,5,1,1,1);

    // 16 = sub2ind(1,1,1)
    ASSERT_EQ(16, sub2ind(0,0,0));
    ASSERT_EQ(17, sub2ind(1,0,0));
    ASSERT_EQ(23, sub2ind(1,2,0));
    ASSERT_EQ(59, sub2ind(1,2,3));
}

TEST_F(SiftTest, Sub2Ind3Test) {

    cudautils::Sub2Ind sub2ind(3,4,5);
    sub2ind.setBasePos(1,1,1);

    // 9 = sub2ind(1,1,1)
    ASSERT_EQ(16, sub2ind(0,0,0));
    ASSERT_EQ(17, sub2ind(1,0,0));
    ASSERT_EQ(23, sub2ind(1,2,0));
    ASSERT_EQ(59, sub2ind(1,2,3));
}

TEST_F(SiftTest, Ind2Sub1Test) {

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

TEST_F(SiftTest, get_grad_ori_vectorTest) {
    const unsigned int x_size = 3;
    const unsigned int y_size = 3;
    const unsigned int z_size = 3;
    const unsigned int keypoint_num = 10000;
    cudautils::SiftParams sift_params;
    double* fv_centers = sift_defaults(&sift_params, x_size,
            y_size, z_size, keypoint_num);

    // create img
    thrust::host_vector<double> img(x_size * y_size * z_size);
    /*thrust::generate(img.begin(), img.end(), rand);*/
    thrust::sequence(img.begin(), img.end());

    unsigned int idx = 14; // center of cube
    unsigned int x_stride = x_size;
    unsigned int y_stride = y_size;
    double vect[3] = {1.0, 0.0, 0.0};
    uint16_t* ix = (uint16_t*) malloc(sift_params.fv_centers_len * sizeof(uint16_t));
    double* yy = (double*) malloc(sift_params.fv_centers_len * sizeof(double));

    double mag = cudautils::get_grad_ori_vector(thrust::raw_pointer_cast(&img[0]), idx, x_stride, y_stride, vect, yy, ix, sift_params, 
            fv_centers);

    print_arr(yy, sift_params.fv_centers_len);

    printf("mag=%f\n", mag);
    for (int i=0; i < sift_params.fv_centers_len; i++) {
        printf("yy[%d]= %f", i, yy[i]);
    }
    logger_->debug("mag={}", mag);
}

TEST_F(SiftTest, SiftFullImageTest) {

    const unsigned int x_size = 2048;
    const unsigned int y_size = 2048;
    const unsigned int z_size = 251;
    const unsigned int keypoint_num = 10000;
    const unsigned int num_gpus = 2;
    const unsigned int num_streams = 20;
    long image_size = x_size * y_size * z_size;
    const unsigned int x_sub_size = min(2048, x_size);
    const unsigned int y_sub_size = min(2048, y_size / num_gpus);
    const unsigned int dx = min(256, x_sub_size);
    const unsigned int dy = min(256, y_sub_size);
    const unsigned int dw = 2;


    cudautils::SiftParams sift_params;
    double* fv_centers = sift_defaults(&sift_params, x_size,
            y_size, z_size, keypoint_num);


    std::shared_ptr<cudautils::Sift> ni =
        std::make_shared<cudautils::Sift>(x_size, y_size, z_size,
                x_sub_size, y_sub_size, dx, dy, dw, num_gpus,
                num_streams, sift_params, fv_centers);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    // create img
    thrust::host_vector<double> img(x_size * y_size * z_size);
    thrust::generate(img.begin(), img.end(), rand);
    /*for (int i=0; i < image_size; i++) {*/
        /*img[i] = rand() % 100 + 1.0;*/
    /*}*/

    // create map
    thrust::host_vector<int8_t> map(x_size * y_size * z_size, 1);
    thrust::fill_n(map.begin(), image_size, 1.0);
    long long idx;
    for (int i=0; i < keypoint_num; i++) {
        // warning not evenly distributed across the image
        idx = (x_size * rand()) % image_size;
        map[idx] = 0.0; // select this point for processing
    }

    ni->setImage(thrust::raw_pointer_cast(img.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    cudautils::Keypoint_store keystore;
    ni->getKeystore(&keystore);

    /*thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);*/
    /*ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));*/

    /*check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);*/
}


}

