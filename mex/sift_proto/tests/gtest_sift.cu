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

TEST_F(SiftTest, DotProductTest) {
    int cols = 3;
    double first[3] = {1,2,3};
    double second[3] = {5, .5, 2};
    double out[1] = {0};
    cudautils::dot_product_wrap<<<1,1>>>(first, second, out, 1, cols);
    ASSERT_EQ(*out, 12);

    // 3x3 matrix
    int rows = 3;
    double matrix[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double vec[3] = {3, 1, 2};
    double* out_arr = (double*) malloc(rows * sizeof(double));
    double answer[3] = {11, 29, 47};
    cudautils::dot_product_wrap<<<1,1>>>(matrix, vec, out_arr, rows, cols);
    for (int i=0; i < rows; i++)
        ASSERT_EQ(out_arr[i], answer[i]);
}

TEST_F(SiftTest, GetBinIdxTest) {

    cudautils::SiftParams sift_params;
    double* fv_centers = sift_defaults(&sift_params, 0, 0, 0, 0);

    double xySpacing = (double) sift_params.xyScale * sift_params.MagFactor;
    double tSpacing = (double) sift_params.tScale * sift_params.MagFactor;

    int xyiradius = round(1.414 * xySpacing * (sift_params.IndexSize + 1) / 2.0);
    int tiradius = round(1.414 * tSpacing * (sift_params.IndexSize + 1) / 2.0);

    // Surrounding radius of pixels are binned for computation 
    // according to sift_params.IndexSize
    int i_bin, j_bin, k_bin;
    for (int i = -xyiradius; i <= xyiradius; i++) {
        for (int j = -xyiradius; j <= xyiradius; j++) {
            for (int k = -tiradius; k <= tiradius; k++) {

                // Find bin idx
                cudautils::get_bin_idx_wrap<<<1,1>>>(i, xyiradius, sift_params.IndexSize, &i_bin);
                cudautils::get_bin_idx_wrap<<<1,1>>>(j, xyiradius, sift_params.IndexSize, &j_bin);
                cudautils::get_bin_idx_wrap<<<1,1>>>(k, tiradius, sift_params.IndexSize, &k_bin);
                // FIXME check correct
                ASSERT_GE(i_bin, 0);
                ASSERT_LT(i_bin, sift_params.IndexSize);
                ASSERT_GE(j_bin, 0);
                ASSERT_LT(j_bin, sift_params.IndexSize);
                ASSERT_GE(k_bin, 0);
                ASSERT_LT(k_bin, sift_params.IndexSize);
            }
        }
    }
}

TEST_F(SiftTest, BinSub2Ind1Test) {
    cudautils::SiftParams sift_params;
    double* fv_centers = sift_defaults(&sift_params, 0,
            0, 0, 0);

    int bin_index;
    for (int i=0; i < sift_params.IndexSize; i++) {
        for (int j=0; j < sift_params.IndexSize; j++) {
            for (int k=0; k < sift_params.IndexSize; k++) {
                for (uint16_t ixi=0; ixi < sift_params.nFaces; ixi++) {
                    cudautils::bin_sub2ind_wrap<<<1,1>>>(i, j, k, ixi, sift_params, &bin_index);
                    ASSERT_LT(bin_index, sift_params.descriptor_len);
                }
            }
        }
    }
    int max = sift_params.descriptor_len - 1;
    int temp;
    cudautils::bin_sub2ind_wrap<<<1,1>>>(1,1,1,79, sift_params, &temp);
    ASSERT_EQ(temp, max);
    cudautils::bin_sub2ind_wrap<<<1,1>>>(0,1,1,79, sift_params, &temp);
    ASSERT_EQ(temp, max - 1);
    cudautils::bin_sub2ind_wrap<<<1,1>>>(1,0,1,79, sift_params, &temp);
    ASSERT_EQ(temp, max - 2);
    cudautils::bin_sub2ind_wrap<<<1,1>>>(1,1,0,79, sift_params, &temp);
    ASSERT_EQ(temp, max - 4);
    cudautils::bin_sub2ind_wrap<<<1,1>>>(1,1,1,78, sift_params, &temp);
    ASSERT_EQ(temp, max - 8);
}


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
    const unsigned int keypoint_num = 1;
    cudautils::SiftParams sift_params;
    double* fv_centers = sift_defaults(&sift_params, x_size,
            y_size, z_size, keypoint_num);
    sift_params.MagFactor = 1;
    sift_params.Tessel_thresh = 1;

    thrust::device_vector<double> device_centers(fv_centers,
            fv_centers + sift_params.fv_centers_len);

    // create img
    thrust::device_vector<double> image(x_size * y_size * z_size);
    thrust::sequence(image.begin(), image.end());

    unsigned long long idx = 14; // center of cube
    int r = 1;
    int c = 1;
    int t = 1;
    unsigned int x_stride = x_size;
    unsigned int y_stride = y_size;
    thrust::device_vector<uint16_t> ix(sift_params.fv_centers_len);
    thrust::device_vector<double> yy(sift_params.fv_centers_len);
    thrust::device_vector<double> vect(3);// = {1.0, 0.0, 0.0};

    double mag; 
    cudautils::get_grad_ori_vector_wrap<<<1,1>>>(thrust::raw_pointer_cast(&image[0]), 
            idx, x_stride, y_stride, r, c, t, thrust::raw_pointer_cast(&vect[0]),
            thrust::raw_pointer_cast(&yy[0]), thrust::raw_pointer_cast(&ix[0]),
            sift_params, thrust::raw_pointer_cast(&device_centers[0]), &mag);
    thrust::host_vector<double> h_yy(yy);

    // make sure yy's values are descending
    double max = h_yy[0];
    for (int i=0; i < sift_params.nFaces; i++) {
        ASSERT_GE(max, h_yy[i]);
        max = h_yy[i];
    }
        
    /*print_arr(thrust::raw_pointer_cast(&h_yy[0]), sift_params.fv_centers_len);*/
}

TEST_F(SiftTest, SiftFullImageTest) {

    const unsigned int x_size = 2048;
    const unsigned int y_size = 2048;
    const unsigned int z_size = 251;
    const unsigned int keypoint_num = 1;
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
    thrust::host_vector<double> image(image_size);
    thrust::generate(image.begin(), image.end(), rand);
    /*for (int i=0; i < image_size; i++) {*/
        /*image[i] = rand() % 100 + 1.0;*/
    /*}*/

    // create map
    thrust::host_vector<int8_t> map(image_size);
    thrust::fill_n(map.begin(), image_size, 1.0);

    int x = 1713;
    int y = 14;
    int z = 14;
    long long idx = x + x_size * y + x_size * y_size * z;
    map[idx] = 0.0; // select this point for processing

    //test a keypoint in the last pixel
    /*map[image_size - 1] = 0.0; //select for processing*/

    /*long long idx;*/
    /*for (int i=0; i < keypoint_num; i++) {*/
        /*// warning not evenly distributed across the image*/
        /*// chosen to roughly distribute across GPUs*/
        /*idx = (x_size * rand()) % image_size;*/
        /*map[idx] = 0.0; // select this point for processing*/
    /*}*/

    ni->setImage(thrust::raw_pointer_cast(image.data()));
    ni->setMapToBeInterpolated(thrust::raw_pointer_cast(map.data()));

    executor.run();

    cudautils::Keypoint_store keystore;
    ni->getKeystore(&keystore);

    /*thrust::host_vector<double> interpolated_image(x_size * y_size * z_size);*/
    /*ni->getImage(thrust::raw_pointer_cast(interpolated_image.data()));*/

    /*check_results(sub2ind, x_size, y_size, z_size, map, img, interpolated_image);*/
}


}

