#include "gtest/gtest.h"

#include "cufftutils.h"
#include <complex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cstdint>
#include <random>
#include <cmath>

class ConvnCufftTest : public ::testing::Test {
protected:
    ConvnCufftTest() {
    }
    virtual ~ConvnCufftTest() {
    }
};

static void initImageVal(float* image, int imageSize, float val) {
    for (int index = 0; index < imageSize; index++) {
        image[index] = val;
    }
}

//Generate uniform numbers [0,1)
static void initImage(float* image, int imageSize) {
    static unsigned seed = 123456789;
    for (int index = 0; index < imageSize; index++) {
        seed = ( 1103515245 * seed + 12345 ) & 0xffffffff;
        image[index] = float(seed)*2.3283064e-10; //2^-32
        //image[index] = (float) index;
        //image[index] = (float) sin(index); //2^-32
        //printf("image(index) %.4f\n", image[index]);
    }
}

void matrix_is_zero(float* first, int* size, bool column_order, int benchmark, float tol)  {
    //convert back to original then check the two matrices
    long long idx;
    for (int i = 0; i<size[0]; ++i) {
        for (int j = 0; j<size[1]; ++j) {
            for (int k = 0; k<size[2]; ++k) {
                idx = cufftutils::convert_idx(i, j, k, size, column_order);
                if (benchmark)
                    printf("idx:%d\n", idx);
                ASSERT_NEAR(first[idx], 0.0, tol);
            }
            if (benchmark)
                printf("\n");
        }
    }
}

void matrix_is_equal_complex(cufftComplex* first, cufftComplex* second, int* size, bool column_order, 
        int benchmark, float tol)  {
    //convert back to original then check the two matrices
    long long idx;
    for (int i = 0; i<size[0]; ++i) {
        for (int j = 0; j<size[1]; ++j) {
            for (int k = 0; k<size[2]; ++k) {
                idx = cufftutils::convert_idx(i, j, k, size, column_order);
                if (benchmark)
                    printf("idx:%d\n", idx);
                ASSERT_NEAR(first[idx].x, second[idx].x, tol);
                ASSERT_NEAR(first[idx].y, second[idx].y, tol);
            }
            if (benchmark)
                printf("\n");
        }
    }
}

void matrix_is_equal(float* first, float* second, int* size, bool column_order, 
        int benchmark, float tol)  {
    //convert back to original then check the two matrices
    long long idx;
    for (int i = 0; i<size[0]; ++i) {
        for (int j = 0; j<size[1]; ++j) {
            for (int k = 0; k<size[2]; ++k) {
                idx = cufftutils::convert_idx(i, j, k, size, column_order);
                if (benchmark)
                    printf("idx=%d (%d, %d, %d): %f %f\n",idx, i, j, k, first[idx], second[idx]);
                ASSERT_NEAR(first[idx], second[idx], tol);
            }
        }
    }
}

TEST_F(ConvnCufftTest, DISABLED_FFTBasicTest) {
    int size[3] = {50, 50, 5};
    int filterdimA[3] = {5, 5, 5};
    bool column_order = 0;
    int N = size[0] * size[1] * size[2];
    int N_kernel = filterdimA[0] * filterdimA[1] * filterdimA[2];
    
    //printf("Initializing cufft sin array\n");
    float* data = new float[N]; 
    float* outArray = new float[N]; 
    float* kernel = new float[N_kernel]; 

    //printf("Sin array created\n");
    for (int i=0; i < N; i++)
        data[i] = sin(i);


    //printf("Initializing kernel\n");
    for (int i=0; i < N_kernel; i++)
        kernel[i] = sin(i);

    //printf("Testing fft\n");
    cufftutils::fft3(data, size, size, outArray, column_order);
}

TEST_F(ConvnCufftTest, DISABLED_PrintDeviceDataTest) {
    long long N = 10;
    long long size_of_data = sizeof(cufftComplex)*N;

    cufftComplex* device_data_input;
    cudaMalloc(&device_data_input, size_of_data);
    cufftComplex *host_data_input = (cufftComplex *)malloc(size_of_data);

    for (int i=0; i < N; i++) {
        host_data_input[i].x = i;
        host_data_input[i].y = i;
    }
    cudaMemcpy(device_data_input, host_data_input, size_of_data, cudaMemcpyHostToDevice);

    cufftutils::printHostData(host_data_input, N);
    cufftutils::printDeviceData(device_data_input, N);

    free(host_data_input);
    cudaFree(device_data_input);
}

TEST_F(ConvnCufftTest, DISABLED_ConvnCompare1GPUTest) {
    int benchmark = 0;
    int size[3] = {50, 50, 5};
    int filterdimA[3] = {2, 2, 2};
    bool column_order = false;
    int algo = 1;
    //float tol = .0001;
    float tol = .000001;
    int N = size[0] * size[1] * size[2];
    int N_kernel = filterdimA[0] * filterdimA[1] * filterdimA[2];
    if (benchmark)
        printf("size %d, %d, %d\n", size[0], size[1], size[2]);
    
    float* hostI = new float[N]; 
    float* hostO = new float[N]; 
    float* hostO_1GPU = new float[N]; 
    float* hostF = new float[N_kernel]; 

    initImageVal(hostI, N, 0.0f);
    initImageVal(hostF, N_kernel, 0.0f);

    if (benchmark)
        printf("mGPU convolution\n");
    cufftutils::conv_handler(hostI, hostF, hostO, algo, size,
            filterdimA, column_order, benchmark);
    if (benchmark)
        printf("Check with 1GPU convolution\n");
    cufftutils::conv_1GPU_handler(hostI, hostF, hostO_1GPU, algo, size,
            filterdimA, column_order, benchmark);

    matrix_is_zero(hostO, size, column_order, 0, tol);
    matrix_is_zero(hostO_1GPU, size, column_order, 0, tol);
    matrix_is_equal(hostO, hostO_1GPU, size, column_order, benchmark, tol);

    if (benchmark)
        printf("Check with sin values");
    initImage(hostI, N);
    initImage(hostF, N_kernel);

    if (benchmark)
        printf("mGPU convolution\n");
    cufftutils::conv_handler(hostI, hostF, hostO, algo, size,
            filterdimA, column_order, benchmark);
    if (benchmark)
        printf("Check with 1GPU convolution\n");
    cufftutils::conv_1GPU_handler(hostI, hostF, hostO_1GPU, algo, size,
            filterdimA, column_order, benchmark);

    matrix_is_equal(hostO, hostO_1GPU, size, column_order, benchmark, tol);
}

TEST_F(ConvnCufftTest, DeviceInitInputsTest) {
    int benchmark = 1;
    //int size[3] = {2, 2, 3};
    int size[3] = {50, 50, 5};
    int filterdimA[3] = {2, 2, 2};
    bool column_order = false;
    int N = size[0] * size[1] * size[2];
    int N_kernel = filterdimA[0] * filterdimA[1] * filterdimA[2];
    
    float* hostI = new float[N]; 
    float* hostF = new float[N_kernel]; 
    float* hostI_column = new float[N]; 
    float* hostF_column = new float[N_kernel]; 

    // Create two random images
    initImage(hostI, N);
    initImage(hostF, N_kernel);

    if (benchmark) {
        printf("\nhostI elements:%d\n", N);
        for (long i = 0; i < N; i++)
           printf("%f\n", hostI[i]);
    }

    if (benchmark)
        printf("Matrix conversions\n");
    cufftutils::convert_matrix(hostI, hostI_column, size, column_order);
    cufftutils::convert_matrix(hostF, hostF_column, filterdimA, column_order);

    int pad_size[3];
    int trim_idxs[3][2];
    cufftutils::get_pad_trim(size, filterdimA, pad_size, trim_idxs);

    if (benchmark) {
        printf("size %d, %d, %d\n", size[0], size[1], size[2]);
        printf("pad_size %d, %d, %d\n", pad_size[0], pad_size[1], pad_size[2]);
        for (int i=0; i < 3; i++) 
            printf("trim_idxs[%d]=%d, %d\n", i, trim_idxs[i][0], trim_idxs[i][1]);
    }
    long long N_padded = pad_size[0] * pad_size[1] * pad_size[2];
    long long size_of_data = N_padded * sizeof(cufftComplex);
    long long float_size = N* sizeof(float);

    if (benchmark)
        printf("cudaMalloc\n");
    cufftComplex* device_data_input,* device_data_kernel,* device_data_input_column,* device_data_kernel_column;
    /*float* devI = new float[N];*/
    /*float* devF = new float[N_kernel];*/
    /*float* devI_column = new float[N];*/
    /*float* devF_column = new float[N_kernel];*/
    cudaMalloc(&device_data_input_column, size_of_data);
    cudaMalloc(&device_data_kernel_column, size_of_data);
    cudaMalloc(&device_data_input, size_of_data);
    cudaMalloc(&device_data_kernel, size_of_data);
    float* devI, *devF, *devI_column, *devF_column;
    cudaMalloc(&devI, float_size);
    cudaMalloc(&devF, float_size);
    cudaMalloc(&devI_column, float_size);
    cudaMalloc(&devF_column, float_size);

    if (benchmark)
        printf("cudaMemcpy\n");
    // Copy the data from 'host' to device using cufftXt formatting
    cudaMemcpy(devI, hostI, float_size, cudaMemcpyHostToDevice);
    cudaMemcpy(devF, hostF, float_size, cudaMemcpyHostToDevice);
    cudaMemcpy(devI_column, hostI_column, float_size, cudaMemcpyHostToDevice);
    cudaMemcpy(devF_column, hostF_column, float_size, cudaMemcpyHostToDevice);

    if (benchmark) {
        printf("\ndevI elements:%d\n", N);
        float *h;

        h = (float *) malloc(float_size);

        cudaMemcpy(h, devI, float_size, cudaMemcpyDeviceToHost);

        for (long long i = 0; i < N; i++) {
            printf("%f\n", h[i]);
        }

        free(h);
    }

    dim3 blockSize(32, 32, 2);
    dim3 gridSize(ceil(pad_size[0] / blockSize.x), ceil(pad_size[1] / blockSize.y), ceil(pad_size[2] / blockSize.z));

    cufftutils::initialize_inputs_par<<<gridSize, blockSize>>>(devI, devF, device_data_input, device_data_kernel, size, pad_size, filterdimA, column_order, benchmark);

    //passing column order should still output c-order data
    cufftutils::initialize_inputs_par<<<gridSize, blockSize>>>(devI_column, devF_column, device_data_input_column,
            device_data_kernel_column, size, pad_size, filterdimA, !column_order, benchmark);

    if (benchmark)
        printf("Copy back to host for error checking");
    // Copy the data from 'host' to device using cufftXt formatting
    cufftComplex *host_data_input = (cufftComplex *)malloc(size_of_data);
    cufftComplex *host_data_kernel = (cufftComplex *)malloc(size_of_data);
    cufftComplex *host_data_input_column = (cufftComplex *)malloc(size_of_data);
    cufftComplex *host_data_kernel_column = (cufftComplex *)malloc(size_of_data);
    cudaMemcpy(host_data_input, device_data_input, size_of_data, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_data_kernel, device_data_kernel, size_of_data, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_data_input_column, device_data_input_column, size_of_data, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_data_kernel_column, device_data_kernel_column, size_of_data, cudaMemcpyDeviceToHost);

    if (benchmark) {
        printf("\nhost_data_input elements:%d\n", N_padded);
        cufftutils::printHostData(host_data_input, N_padded);
        printf("\nhost_data_kernel elements:%d\n", N_padded);
        cufftutils::printHostData(host_data_kernel, N_padded);

        printf("\nhost_data_input_column elements:%d\n", N_padded);
        cufftutils::printHostData(host_data_input_column, N_padded);
        printf("\nhost_data_kernel_column elements:%d\n", N_padded);
        cufftutils::printHostData(host_data_kernel_column, N_padded);
    }

    // Check that initialize inputs put both row and column ordered matrix into c-order
    matrix_is_equal_complex(host_data_input, host_data_input_column, size, 
            column_order, benchmark, 0.0); //tol=0.0 means exactly equal
    matrix_is_equal_complex(host_data_kernel, host_data_kernel_column, size, 
            column_order, benchmark, 0.0);

    // test padding is correct for c-order
    long long idx;
    long long pad_idx;
    long long idx_filter;
    for (int i = 0; i<pad_size[0]; ++i) {
        for (int j = 0; j<pad_size[1]; ++j) {
            for (int k = 0; k<pad_size[2]; ++k) {
                idx = cufftutils::convert_idx(i, j, k, size, column_order);
                pad_idx = cufftutils::convert_idx(i, j, k, pad_size, column_order);
                idx_filter = cufftutils::convert_idx(i, j, k, filterdimA, column_order);

                if (benchmark) {
                    printf("pad_idx=%d idx=%d (%d %d %d) %d | ",
                            pad_idx, idx, i, j, k, (int) host_data_input[pad_idx].x);
                }

                if ((i < size[0]) && (j < size[1]) && (k < size[2]) ) {
                    ASSERT_EQ(host_data_input[pad_idx].x, hostI[idx]);
                } else {
                    ASSERT_EQ(host_data_input[pad_idx].x, 0.0f);
                }
                ASSERT_EQ(host_data_input[pad_idx].y, 0.0f);

                if ((i < filterdimA[0]) && (j < filterdimA[1]) && (k < filterdimA[2])) {
                    ASSERT_EQ(host_data_kernel[pad_idx].x, hostF[idx_filter]);
                } else {
                    ASSERT_EQ(host_data_kernel[pad_idx].x, 0.0f);
                }
                ASSERT_EQ(host_data_kernel[pad_idx].y, 0.0f);

            }
            if (benchmark)
                printf("\n");
        }
    }
    delete[] hostI;
    delete[] hostF;
    free(host_data_input);
    free(host_data_kernel);
}

TEST_F(ConvnCufftTest, InitializePadTest) {
    int benchmark = 0;
    //int size[3] = {2, 2, 3};
    int size[3] = {50, 50, 5};
    int filterdimA[3] = {2, 2, 2};
    bool column_order = false;
    int N = size[0] * size[1] * size[2];
    int N_kernel = filterdimA[0] * filterdimA[1] * filterdimA[2];
    
    float* hostI = new float[N]; 
    float* hostF = new float[N_kernel]; 
    float* hostI_column = new float[N]; 
    float* hostF_column = new float[N_kernel]; 


    // Create two random images
    initImage(hostI, N);
    initImage(hostF, N_kernel);

    if (benchmark)
        printf("Matrix conversions\n");
    cufftutils::convert_matrix(hostI, hostI_column, size, column_order);
    cufftutils::convert_matrix(hostF, hostF_column, filterdimA, column_order);

    int pad_size[3];
    int trim_idxs[3][2];
    cufftutils::get_pad_trim(size, filterdimA, pad_size, trim_idxs);

    // test pad size and trim
    for (int i=0; i < 3; i++) {
        ASSERT_EQ((size[i] + filterdimA[i] - 1), pad_size[i] ) ;
        // check for mem alloc issues
        ASSERT_EQ((trim_idxs[i][1] - trim_idxs[i][0]), size[i] ) ;
    }

    if (benchmark) {
        printf("size %d, %d, %d\n", size[0], size[1], size[2]);
        printf("pad_size %d, %d, %d\n", pad_size[0], pad_size[1], pad_size[2]);
        for (int i=0; i < 3; i++) 
            printf("trim_idxs[%d]=%d, %d\n", i, trim_idxs[i][0], trim_idxs[i][1]);
    }
    long long N_padded = pad_size[0] * pad_size[1] * pad_size[2];
    long long size_of_data = N_padded * sizeof(cufftComplex);

    cufftComplex *host_data_input = (cufftComplex *)malloc(size_of_data);
    cufftComplex *host_data_kernel = (cufftComplex *)malloc(size_of_data);
    cufftComplex *host_data_input_column = (cufftComplex *)malloc(size_of_data);
    cufftComplex *host_data_kernel_column = (cufftComplex *)malloc(size_of_data);

    cufftutils::initialize_inputs(hostI, hostF, host_data_input, host_data_kernel,
            size, pad_size, filterdimA, column_order);

    //passing column order should still output c-order data
    cufftutils::initialize_inputs(hostI_column, hostF_column, host_data_input_column,
            host_data_kernel_column, size, pad_size, filterdimA, !column_order);

    if (benchmark) {
        printf("\nhost_data_input elements:%d\n", N_padded);
        cufftutils::printHostData(host_data_input, N_padded);
        printf("\nhost_data_kernel elements:%d\n", N_padded);
        cufftutils::printHostData(host_data_kernel, N_padded);

        printf("\nhost_data_input_column elements:%d\n", N_padded);
        cufftutils::printHostData(host_data_input_column, N_padded);
        printf("\nhost_data_kernel_column elements:%d\n", N_padded);
        cufftutils::printHostData(host_data_kernel_column, N_padded);
    }

    // Check that initialize inputs put both row and column ordered matrix into c-order
    matrix_is_equal_complex(host_data_input, host_data_input_column, size, 
            column_order, benchmark, 0.0);
    matrix_is_equal_complex(host_data_kernel, host_data_kernel, size, 
            column_order, benchmark, 0.0);

    // test padding is correct for c-order
    long long idx;
    long long pad_idx;
    long long idx_filter;
    for (int i = 0; i<pad_size[0]; ++i) {
        for (int j = 0; j<pad_size[1]; ++j) {
            for (int k = 0; k<pad_size[2]; ++k) {
                idx = cufftutils::convert_idx(i, j, k, size, column_order);
                pad_idx = cufftutils::convert_idx(i, j, k, pad_size, column_order);
                idx_filter = cufftutils::convert_idx(i, j, k, filterdimA, column_order);

                if (benchmark) {
                    printf("pad_idx=%d idx=%d (%d %d %d) %d | ",
                            pad_idx, idx, i, j, k, (int) host_data_input[pad_idx].x);
                }

                if ((i < size[0]) && (j < size[1]) && (k < size[2]) ) {
                    ASSERT_EQ(host_data_input[pad_idx].x, hostI[idx]);
                } else {
                    ASSERT_EQ(host_data_input[pad_idx].x, 0.0f);
                }
                ASSERT_EQ(host_data_input[pad_idx].y, 0.0f);

                if ((i < filterdimA[0]) && (j < filterdimA[1]) && (k < filterdimA[2])) {
                    ASSERT_EQ(host_data_kernel[pad_idx].x, hostF[idx_filter]);
                } else {
                    ASSERT_EQ(host_data_kernel[pad_idx].x, 0.0f);
                }
                ASSERT_EQ(host_data_kernel[pad_idx].y, 0.0f);

            }
            if (benchmark)
                printf("\n");
        }
    }
    delete[] hostI;
    delete[] hostF;
    free(host_data_input);
    free(host_data_kernel);
}

TEST_F(ConvnCufftTest, DISABLED_1GPUConvnFullImageTest) {
    //int size[3] = {126, 1024, 1024}; 
    int size[3] = {1024, 1024, 126};
    int filterdimA[3] = {5, 5, 5};
    int benchmark = 0;
    bool column_order = false;
    int algo = 1;
    float tol = .0001;
    long long N = size[0] * size[1] * size[2];
    long long N_kernel = filterdimA[0] * filterdimA[1] * filterdimA[2];
    
    float* data = new float[N]; 
    float* kernel = new float[N_kernel]; 

    initImage(data, N); //random input matrix
    initImageVal(kernel, N_kernel, 0.0); //kernel of zeros

    cufftutils::conv_1GPU_handler(data, kernel, data, algo, size,
            filterdimA, column_order, benchmark);

    matrix_is_zero(data, size, column_order, benchmark, tol);
}

TEST_F(ConvnCufftTest, DISABLED_ConvnFullImageTest) {
    //int size[3] = {2048, 2048, 141};
    //int size[3] = {141, 2048, 2048};
    int size[3] = {50, 50, 5};
    int filterdimA[3] = {5, 5, 5};
    int benchmark = 0;
    bool column_order = false;
    int algo = 1;
    float tol = .0001;
    long long N = size[0] * size[1] * size[2];
    long long N_kernel = filterdimA[0] * filterdimA[1] * filterdimA[2];
    
    float* data = new float[N]; 
    float* kernel = new float[N_kernel]; 

    initImage(data, N); //random input matrix
    initImageVal(kernel, N_kernel, 0.0); //kernel of zeros

    cufftutils::conv_handler(data, kernel, data, algo, size,
            filterdimA, column_order, benchmark);

    matrix_is_zero(data, size, column_order, benchmark, tol);
}

TEST_F(ConvnCufftTest, DISABLED_ConvnColumnOrderingTest) {

    // generate params
    int benchmark = 0;
    float tol = .8;
    int algo = 0;
    bool column_order = false;
    int size[3] = {50, 50, 5};
    int filterdimA[] = {2, 2, 2};
    int filtersize = filterdimA[0]*filterdimA[1]*filterdimA[2];
    int insize = size[0]*size[1]*size[2];

    // Create a random filter and image
    float* hostI;
    float* hostF;
    float* hostI_column;
    float* hostF_column;
    float* hostO;
    float* hostO_column;

    hostI = new float[insize];
    hostF = new float[filtersize];
    hostI_column = new float[insize];
    hostF_column = new float[filtersize];
    hostO = new float[insize];
    hostO_column = new float[insize];

    // Create two random images
    initImage(hostI, insize);
    initImage(hostF, filtersize);

    if (benchmark)
        printf("Matrix conversions\n");
    cufftutils::convert_matrix(hostI, hostI_column, size, column_order);
    cufftutils::convert_matrix(hostF, hostF_column, filterdimA, column_order);

    if (benchmark) {
        printf("\nhostF elements:%d\n", filtersize);
        for (int i = 0; i < filtersize; i++)
            printf("%.1f\n", hostF[i]);
        printf("\nhostF_column elements:%d\n", filtersize);
        for (int i = 0; i < filtersize; i++)
            printf("%.1f\n", hostF_column[i]);
    }

    if (benchmark)
        printf("\n\noriginal order:%d\n", column_order);
    cufftutils::conv_handler(hostI, hostF, hostO, algo, size, 
            filterdimA, column_order, benchmark);

    if (benchmark)
        printf("\n\ntest with column_order\n", !column_order);
    cufftutils::conv_handler(hostI_column, hostF_column, hostO_column, 
            algo, size, filterdimA, !column_order, benchmark);

    //convert back to original then check the two matrices
    long long idx;
    long long col_idx;
    float val;
    for (int i = 0; i<size[0]; ++i) {
        for (int j = 0; j<size[1]; ++j) {
            for (int k = 0; k<size[2]; ++k) {
                idx = cufftutils::convert_idx(i, j, k, size, column_order);
                col_idx = cufftutils::convert_idx(i, j, k, size, !column_order);
                if (benchmark) {
                    val = hostO_column[col_idx]; // get the real component
                    printf("idx=%d (%d, %d, %d): %f | ", idx, i, j, k, val);
                }
                ASSERT_NEAR(hostO[idx], hostO_column[col_idx], tol);
            }
            if (benchmark)
                printf("\n");
        }
    }


}

