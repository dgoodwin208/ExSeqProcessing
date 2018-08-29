#include "gtest/gtest.h"

#include "sift.h"
#include "cuda_task_executor.h"
#include "gpudevice.h"
#include <vector>
#include <iterator>
#include <algorithm>
#include <cublas_v2.h>
#include <stdexcept>

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
        printf("print_arr\n");
        for (int i=0; i < N; i++) {
            logger_->debug("\t[{}]={}", i, arr[i]);
            /*printf("\t[%d]=%.5f\n", i, arr[i]);*/
        }
    }

};

TEST_F(SiftTest, DotProductTest) {
    int cols = 3;
    double first[3] = {1,2,3};
    double *dfirst;
    cudaMalloc(&dfirst, sizeof(double) * cols);
    cudaMemcpy(dfirst, first, sizeof(double) * cols, cudaMemcpyHostToDevice);

    double second[3] = {5, .5, 2};
    double* dsecond;
    cudaMalloc(&dsecond, sizeof(double) * cols);
    cudaMemcpy(dsecond, second, sizeof(double) * cols, cudaMemcpyHostToDevice);

    double out[1] = {0};
    double *dout;
    cudaMalloc(&dout, sizeof(double));
    cudautils::dot_product_wrap<<<1,1>>>(dfirst, dsecond, dout, 1, cols);
    cudaMemcpy(out, dout, sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_EQ(*out, 12);

    // 3x3 matrix
    int rows = 3;
    double matrix[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double *dmatrix;
    cudaMalloc(&dmatrix, sizeof(double) * 9);
    cudaMemcpy(dmatrix, matrix, sizeof(double) * 9, cudaMemcpyHostToDevice);

    double vec[3] = {3, 1, 2};
    double *dvec;
    cudaMalloc(&dvec, sizeof(double) * 3);
    cudaMemcpy(dvec, vec, sizeof(double) * 3, cudaMemcpyHostToDevice);

    double* dout_arr;
    cudaMalloc(&dout_arr, rows * sizeof(double));
    cudautils::dot_product_wrap<<<1,1>>>(dmatrix, dvec, dout_arr, rows, cols);
    double* out_arr = (double*) malloc(rows * sizeof(double));
    cudaMemcpy(out_arr, dout_arr, sizeof(double) * rows, cudaMemcpyDeviceToHost);
    double answer[3] = {11, 29, 47};
    for (int i=0; i < rows; i++)
        ASSERT_EQ(out_arr[i], answer[i]);

    cudaFree(dfirst);
    cudaFree(dsecond);
    cudaFree(dout);
    cudaFree(dmatrix);
    cudaFree(dvec);
    cudaFree(dout_arr);
    free(out_arr);
}

TEST_F(SiftTest, GetBinIdxTest) {

    cudautils::SiftParams sift_params;
    double* fv_centers = sift_defaults(&sift_params, 0, 0, 0, 0);
    const int IndexSize = sift_params.IndexSize;

    double xySpacing = (double) sift_params.xyScale * sift_params.MagFactor;
    double tSpacing = (double) sift_params.tScale * sift_params.MagFactor;

    int xyiradius = round(1.414 * xySpacing * (sift_params.IndexSize + 1) / 2.0);
    int tiradius = round(1.414 * tSpacing * (sift_params.IndexSize + 1) / 2.0);

    // Surrounding radius of pixels are binned for computation 
    // according to sift_params.IndexSize
    int *i_bin, *j_bin, *k_bin;
    int hi, hj, hk;
    cudaMalloc(&i_bin, sizeof(int));
    cudaMalloc(&j_bin, sizeof(int));
    cudaMalloc(&k_bin, sizeof(int));
    for (int i = -xyiradius; i <= xyiradius; i++) {
        for (int j = -xyiradius; j <= xyiradius; j++) {
            for (int k = -tiradius; k <= tiradius; k++) {

                // Find bin idx
                cudautils::get_bin_idx_wrap<<<1,1>>>(i, xyiradius, IndexSize, i_bin);
                cudautils::get_bin_idx_wrap<<<1,1>>>(j, xyiradius, IndexSize, j_bin);
                cudautils::get_bin_idx_wrap<<<1,1>>>(k, tiradius, IndexSize, k_bin);

                cudaMemcpy(&hi, i_bin, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&hj, j_bin, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&hk, k_bin, sizeof(int), cudaMemcpyDeviceToHost);
                ASSERT_GE(hi, 0);
                ASSERT_LT(hi, sift_params.IndexSize);
                ASSERT_GE(hj, 0);
                ASSERT_LT(hj, sift_params.IndexSize);
                ASSERT_GE(hk, 0);
                ASSERT_LT(hk, sift_params.IndexSize);
            }
        }
    }
    cudaFree(i_bin);
    cudaFree(j_bin);
    cudaFree(k_bin);
    free(fv_centers);
}

TEST_F(SiftTest, BinSub2Ind1Test) {
    cudautils::SiftParams sift_params;
    double* fv_centers = sift_defaults(&sift_params, 0,
            0, 0, 0);

    int* bin_index;
    cudaMalloc(&bin_index, sizeof(int));
    int temp;
    for (int i=0; i < sift_params.IndexSize; i++) {
        for (int j=0; j < sift_params.IndexSize; j++) {
            for (int k=0; k < sift_params.IndexSize; k++) {
                for (uint16_t ixi=0; ixi < sift_params.nFaces; ixi++) {
                    cudautils::bin_sub2ind_wrap<<<1,1>>>(i, j, k, ixi, sift_params, bin_index);
                    cudaMemcpy(&temp, bin_index, sizeof(int), cudaMemcpyDeviceToHost);
                    ASSERT_LT(temp, sift_params.descriptor_len);
                }
            }
        }
    }

    int max = sift_params.descriptor_len - 1;

    cudautils::bin_sub2ind_wrap<<<1,1>>>(1,1,1,79, sift_params, bin_index);
    cudaMemcpy(&temp, bin_index, sizeof(int), cudaMemcpyDeviceToHost);
    ASSERT_EQ(temp, max);

    cudautils::bin_sub2ind_wrap<<<1,1>>>(0,1,1,79, sift_params, bin_index);
    cudaMemcpy(&temp, bin_index, sizeof(int), cudaMemcpyDeviceToHost);
    ASSERT_EQ(temp, max - 1);

    cudautils::bin_sub2ind_wrap<<<1,1>>>(1,0,1,79, sift_params, bin_index);
    cudaMemcpy(&temp, bin_index, sizeof(int), cudaMemcpyDeviceToHost);
    ASSERT_EQ(temp, max - 2);

    cudautils::bin_sub2ind_wrap<<<1,1>>>(1,1,0,79, sift_params, bin_index);
    cudaMemcpy(&temp, bin_index, sizeof(int), cudaMemcpyDeviceToHost);
    ASSERT_EQ(temp, max - 4);

    cudautils::bin_sub2ind_wrap<<<1,1>>>(1,1,1,78, sift_params, bin_index);
    cudaMemcpy(&temp, bin_index, sizeof(int), cudaMemcpyDeviceToHost);
    ASSERT_EQ(temp, max - 8);

    cudaFree(bin_index);
    free(fv_centers);
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
    long long N = x_size * y_size * z_size;
    const unsigned int keypoint_num = 1;
    cudautils::SiftParams sift_params;
    double* fv_centers = sift_defaults(&sift_params, x_size,
            y_size, z_size, keypoint_num);
    sift_params.MagFactor = 1;
    sift_params.Tessel_thresh = 1;

    double *image, *h_image;
    // create img
    h_image = (double*) malloc(sizeof(double) * N);
    cudaMalloc(&image, sizeof(double) * N);
    for (int i=0; i < N; i++) {
        h_image[i] = i;
    }
    cudaMemcpy(image, h_image, N * sizeof(double), cudaMemcpyHostToDevice);

    double* device_centers;
    // sift_params.fv_centers must be placed on device since array passed to cuda kernel
    // default fv_centers_len 80 * 3 (3D) = 240;
    cudaSafeCall(cudaMalloc((void **) &device_centers,
                sizeof(double) * sift_params.fv_centers_len));
    cudaSafeCall(cudaMemcpy((void *) device_centers, (const void *) fv_centers,
            (size_t) sizeof(double) * sift_params.fv_centers_len,
            cudaMemcpyHostToDevice));
    
    unsigned long long idx = 14; // center of cube
    int r = 1;
    int c = 1;
    int t = 1;
    unsigned int x_stride = x_size;
    unsigned int y_stride = y_size;
    double* yy, *h_yy;
    cudaMalloc(&yy, sizeof(double) * sift_params.fv_centers_len);
    h_yy = (double*) malloc(sizeof(double) * sift_params.fv_centers_len);
    uint16_t* ix;
    cudaMalloc(&ix, sizeof(uint16_t) * sift_params.fv_centers_len);
    double vect[3] = {1.0, 0.0, 0.0};
    double *dvect;
    cudaMalloc(&dvect, sizeof(double) * 3);
    cudaMemcpy(dvect, vect, sizeof(double) * 3, cudaMemcpyHostToDevice);

    double mag; 
    cudautils::get_grad_ori_vector_wrap<<<1,1>>>(image,
            idx, x_stride, y_stride, r, c, t, dvect, yy, ix, sift_params,
            device_centers, &mag);
    cudaMemcpy(h_yy, yy, sizeof(double) * sift_params.fv_centers_len, cudaMemcpyDeviceToHost);

    // make sure yy's values are descending
    double max = h_yy[0];
    for (int i=1; i < sift_params.nFaces; i++) {
        ASSERT_GE(max, h_yy[i]);
        max = h_yy[i]; //update
    }
        
    free(fv_centers);
}

TEST_F(SiftTest, SiftBoundaryTest) {

    const unsigned int x_size = 10;
    const unsigned int y_size = 10;
    const unsigned int z_size = 6;
    long long N = x_size * y_size * z_size;
    const unsigned int keypoint_num = 1;
    const unsigned int num_gpus = 2;
    const unsigned int num_streams = 1;
    const unsigned int x_sub_size = x_size;
    const unsigned int y_sub_size = y_size / num_gpus;
    const unsigned int dx = x_sub_size;
    const unsigned int dy = y_sub_size;
    const unsigned int dw = 0;

    cudautils::SiftParams sift_params;
    double* fv_centers = sift_defaults(&sift_params, x_size,
            y_size, z_size, keypoint_num);

    /*// create img*/
    /*double *image, *h_image; */
    /*int8_t *map, *h_map;*/
    /*h_image = (double*) malloc(sizeof(double) * N);*/
    /*h_map = (int8_t*) malloc(sizeof(int8_t) * N);*/
    /*cudaMalloc(&image, sizeof(double) * N);*/
    /*cudaMalloc(&map, sizeof(int8_t) * N);*/
    /*for (int i=0; i < N; i++) {*/
        /*h_image[i] = i;*/
        /*h_map[i] = 1;*/
    /*}*/

    // create img
    thrust::host_vector<double> image(N);
    thrust::generate(image.begin(), image.end(), rand);

    // create map
    thrust::host_vector<int8_t> map(N);
    thrust::fill_n(map.begin(), N, 1.0);

    // test a keypoint in the last pixel
    map[N - 1] = 0.0; //select for processing

    // investigate a boundary point between GPUs
    int x = 5;
    int y = 5;
    int z = 3;
    long long idx = x + x_size * y + x_size * y_size * z;
    map[idx] = 0.0; // select this point for processing

    /*cudaMemcpy(image, h_image, N * sizeof(double), cudaMemcpyHostToDevice);*/
    /*cudaMemcpy(map, h_map, N * sizeof(int8_t), cudaMemcpyHostToDevice);*/

    std::shared_ptr<cudautils::Sift> ni =
        std::make_shared<cudautils::Sift>(x_size, y_size, z_size,
                x_sub_size, y_sub_size, dx, dy, dw, num_gpus,
                num_streams, sift_params, fv_centers);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    ni->setImage(image.data());
    ni->setMap(map.data());

    executor.run();

    free(fv_centers);
}

TEST_F(SiftTest, SiftSimpleTest) {

    const unsigned int x_size = 10;
    const unsigned int y_size = 10;
    const unsigned int z_size = 6;
    const unsigned int keypoint_num = 1;
    const unsigned int num_gpus = 1;
    const unsigned int num_streams = 1;
    long image_size = x_size * y_size * z_size;
    const unsigned int x_sub_size = x_size;
    const unsigned int y_sub_size = y_size / num_gpus;
    const unsigned int dx = x_sub_size;
    const unsigned int dy = y_sub_size;
    const unsigned int dw = 0;

    cudautils::SiftParams sift_params;
    double* fv_centers = sift_defaults(&sift_params, x_size,
            y_size, z_size, keypoint_num);

    // create img
    thrust::host_vector<double> image(image_size);
    thrust::generate(image.begin(), image.end(), rand);

    // create map
    thrust::host_vector<int8_t> map(image_size);
    thrust::fill_n(map.begin(), image_size, 1.0);

    long long idx;
    for (int i=0; i < keypoint_num; i++) {
        // warning not evenly distributed across the image
        // chosen to roughly distribute across GPUs (y axis)
        idx = (x_size * rand()) % image_size;
        map[idx] = 0.0; // select this point for processing
    }

    std::shared_ptr<cudautils::Sift> ni =
        std::make_shared<cudautils::Sift>(x_size, y_size, z_size,
                x_sub_size, y_sub_size, dx, dy, dw, num_gpus,
                num_streams, sift_params, fv_centers);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    ni->setImage(thrust::raw_pointer_cast(image.data()));
    ni->setMap(thrust::raw_pointer_cast(map.data()));

    executor.run();

    free(fv_centers);
}

// Check output with saved binary 
TEST_F(SiftTest, CheckBinaryTest) {

        std::string in_image_filename1("test_img.bin");
        std::string in_map_filename2("test_map.bin");
        std::string out_filename("gtest_output.bin");
        // Compares to test_output.bin file in tests/

        unsigned int x_size, y_size, z_size, x_size1, y_size1, z_size1;

        std::ifstream fin1(in_image_filename1, std::ios::binary);
        if (fin1.is_open()) {
            fin1.read((char*)&x_size, sizeof(unsigned int));
            fin1.read((char*)&y_size, sizeof(unsigned int));
            fin1.read((char*)&z_size, sizeof(unsigned int));
        } else { 
            throw std::invalid_argument( "Unable to open or find file: `test_img.bin` in current directory");
        }

        std::ifstream fin2(in_map_filename2, std::ios::binary);
        if (fin2.is_open()) {
            fin2.read((char*)&x_size1, sizeof(unsigned int));
            fin2.read((char*)&y_size1, sizeof(unsigned int));
            fin2.read((char*)&z_size1, sizeof(unsigned int));
        } else { 
            throw std::invalid_argument( "Unable to open or find file: `test_map.bin` in current directory");
        }

        if (x_size != x_size1 || y_size != y_size1 || z_size != z_size1) {
            logger_->error("the dimension of image and map is not the same. image({},{},{}), map({},{},{})",
                    x_size, y_size, z_size, x_size1, y_size1, z_size1);
            fin1.close();
            fin2.close();
            /*ASSERT_TRUE(false);*/
            throw std::invalid_argument("Dimension of image and map is not the same");
        }

        std::vector<double> in_image(x_size * y_size * z_size);
        std::vector<int8_t> in_map  (x_size * y_size * z_size);
        fin1.read((char*)in_image.data(), x_size * y_size * z_size * sizeof(double));
        fin2.read((char*)in_map.data(), x_size * y_size * z_size * sizeof(int8_t));
        fin1.close();
        fin2.close();

        cudautils::SiftParams sift_params;
        double* fv_centers = sift_defaults(&sift_params,
                x_size, y_size, z_size, 0);
        int stream_num = 20;
        int x_substream_stride = 256;
        int y_substream_stride = 256;
        
        int num_gpus = cudautils::get_gpu_num();
        logger_->info("# of gpus = {}", num_gpus);
        logger_->info("# of streams = {}", stream_num);

        const unsigned int x_sub_size = x_size;
        const unsigned int y_sub_size = y_size / num_gpus;
        const unsigned int dw = 0;

        const unsigned int dx = min(x_substream_stride, x_sub_size);
        const unsigned int dy = min(y_substream_stride, y_sub_size);

        std::shared_ptr<cudautils::Sift> ni =
            std::make_shared<cudautils::Sift>(x_size, y_size, z_size,
                    x_sub_size, y_sub_size, dx, dy, dw, num_gpus,
                    stream_num, sift_params, fv_centers);

        cudautils::CudaTaskExecutor executor(num_gpus, stream_num, ni);

        ni->setImage(thrust::raw_pointer_cast(in_image.data()));
        ni->setMap(thrust::raw_pointer_cast(in_map.data()));

        executor.run();

        cudautils::Keypoint_store keystore;
        ni->getKeystore(&keystore);

        FILE* pFile = fopen(out_filename.c_str(), "w");
        if (pFile != NULL) {
            // print keystore 
            for (int i=0; i < keystore.len; i++) {
                cudautils::Keypoint key = keystore.buf[i];
                fprintf(pFile, "Keypoint:%d\n", i);
                for (int j=0; j < sift_params.descriptor_len; j++) {
                    fprintf(pFile, "\t%d: %d\n", j, (int) key.ivec[j]);
                }
            }
            fclose(pFile);
        } else { 
            throw std::invalid_argument( "Unable to open output file\nMake sure `test_output.bin` is in current dir: tests/");
        }
 
        ASSERT_EQ(system("diff test_output.bin gtest_output.bin"), 0);
        free(fv_centers);
}

// Disabled to speed up quick test runs
TEST_F(SiftTest, SiftFullImageTest) {

    const unsigned int x_size = 1024;
    const unsigned int y_size = 1024;
    const unsigned int z_size = 126;
    const unsigned int keypoint_num = 1;
    const unsigned int num_gpus = 2;
    const unsigned int num_streams = 20;
    long image_size = x_size * y_size * z_size;
    const unsigned int x_sub_size = x_size;
    const unsigned int y_sub_size = y_size / num_gpus;
    const unsigned int dx = 256;
    const unsigned int dy = 256;
    const unsigned int dw = 0;

    cudautils::SiftParams sift_params;
    double* fv_centers = sift_defaults(&sift_params, x_size,
            y_size, z_size, keypoint_num);

    // create img
    thrust::host_vector<double> image(image_size);
    thrust::generate(image.begin(), image.end(), rand);

    // create map
    thrust::host_vector<int8_t> map(image_size);
    thrust::fill_n(map.begin(), image_size, 1.0);

    long long idx;
    for (int i=0; i < keypoint_num; i++) {
        // warning not evenly distributed across the image
        // chosen to roughly distribute across GPUs
        idx = (x_size * rand()) % image_size;
        map[idx] = 0.0; // select this point for processing
    }

    std::shared_ptr<cudautils::Sift> ni =
        std::make_shared<cudautils::Sift>(x_size, y_size, z_size,
                x_sub_size, y_sub_size, dx, dy, dw, num_gpus,
                num_streams, sift_params, fv_centers);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);


    ni->setImage(thrust::raw_pointer_cast(image.data()));
    ni->setMap(thrust::raw_pointer_cast(map.data()));

    executor.run();

    cudautils::Keypoint_store keystore;
    ni->getKeystore(&keystore);

    free(fv_centers);
}


}

