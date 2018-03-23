#include "gtest/gtest.h"

#include "cufftutils.h"
#include "cuda3Dutils.h"
#include <complex.h>
#include <cufft.h>
#include <vector>
#include <cstdint>
#include <random>

namespace {

class ConvnCufftTest : public ::testing::Test {
protected:
    ConvnCufftTest() {
    }
    virtual ~ConvnCufftTest() {
    }
};

//Generate uniform numbers [0,1)
static void initImage(float* image, int imageSize) {
    static unsigned seed = 123456789;
    for (int index = 0; index < imageSize; index++) {
        seed = ( 1103515245 * seed + 12345 ) & 0xffffffff;
        image[index] = float(seed)*2.3283064e-10; //2^-32
    }
}

TEST_F(ConvnCufftTest, DISABLED_FFTBasicTest) {
    int size[3] = {50, 50, 10};
    int filterdimA[3] = {5, 5, 5};
    int benchmark = 1;
    int pad = 1;
    int algo = 1;
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

TEST_F(ConvnCufftTest, ConvnCompareTest) {
    int size[3] = {200, 200, 300};
    int filterdimA[3] = {2, 2, 2};
    int benchmark = 1;
    bool column_order = false;
    int algo = 1;
    int result = 0;
    int N = size[0] * size[1] * size[2];
    int N_kernel = filterdimA[0] * filterdimA[1] * filterdimA[2];
    printf("size %d, %d, %d\n", size[0], size[1], size[2]);
    long long size_of_data = N* sizeof(cufftComplex);
    
    float* hostI = new float[N]; 
    float* hostO = new float[N]; 
    float* hostF = new float[N_kernel]; 

    cufftComplex *host_data_input = (cufftComplex *)malloc(size_of_data);
    if (!host_data_input) { printf("malloc input failed"); }
    cufftComplex *host_data_kernel = (cufftComplex *)malloc(size_of_data);
    if (!host_data_kernel) { printf("malloc kernel failed"); }

    for (int i=0; i < N; i++) {
        hostI[i] = (float) sin(i);
        host_data_input[i].x = hostI[i];
        host_data_input[i].y = 0;
    }

    for (int i=0; i < N_kernel; i++) {
        hostF[i] = (float) sin(i);
        host_data_kernel[i].x = hostF[i];
        host_data_kernel[i].y = 0;
    }

    printf("Testing convolution\n");
    cufftutils::conv_handler(hostI, hostF, hostO, algo, size,
            filterdimA, column_order, benchmark);
    cuda3Dutils::cudaConvolution3D(host_data_input, host_data_kernel, size, filterdimA, 
            32, 256);

    //convert back to original then check the two matrices
    long long idx;
    for (int i = 0; i<size[0]; ++i) {
        for (int j = 0; j<size[1]; ++j) {
            for (int k = 0; k<size[2]; ++k) {
                idx = cufftutils::convert_idx(i, j, k, size, column_order);
                //val = host_data_input[pad_idx].x; // get the real component
                //printf("idx=%d (%d, %d, %d): %d | ",idx, i, j, k, (int) val);
                EXPECT_NEAR(hostO[idx], host_data_input[idx].x, 1.0);
            }
            //printf("\n");
        }
    }
}

TEST_F(ConvnCufftTest, InitializePadTest) {
    int size[3] = {2, 2, 3};
    int filterdimA[3] = {2, 2, 2};
    int benchmark = 1;
    bool column_order = false;
    int algo = 1;
    int result = 0;
    int N = size[0] * size[1] * size[2];
    int N_kernel = filterdimA[0] * filterdimA[1] * filterdimA[2];
    
    float* hostI = new float[N]; 
    float* hostF = new float[N_kernel]; 

    for (int i=0; i < N; i++)
        hostI[i] = (float) i;

    for (int i=0; i < N_kernel; i++)
        hostF[i] = (float) i;

    int pad_size[3];
    int trim_idxs[3][2];
    cufftutils::get_pad_trim(size, filterdimA, pad_size, trim_idxs);

    // test pad size and trim
    for (int i=0; i < 3; i++) {
        ASSERT_EQ((size[i] + filterdimA[i] - 1), pad_size[i] ) ;
        // check for mem alloc issues
        ASSERT_EQ((trim_idxs[i][1] - trim_idxs[i][0]), size[i] ) ;
            //{ printf("Error in same size output calculation first: %d, last: %d\n",
                    //trim_idxs[i][0], trim_idxs[i][1]); exit(EXIT_FAILURE); } 
    }

    printf("size %d, %d, %d\n", size[0], size[1], size[2]);
    printf("pad_size %d, %d, %d\n", pad_size[0], pad_size[1], pad_size[2]);
    long long N_padded = pad_size[0] * pad_size[1] * pad_size[2];
    long long size_of_data = N_padded * sizeof(cufftComplex);

    cufftComplex *host_data_input = (cufftComplex *)malloc(size_of_data);
    if (!host_data_input) { printf("malloc input failed"); }
    cufftComplex *host_data_kernel = (cufftComplex *)malloc(size_of_data);
    if (!host_data_kernel) { printf("malloc kernel failed"); }

    cufftutils::initialize_inputs(hostI, hostF, host_data_input, host_data_kernel, size, pad_size, filterdimA, column_order);

    // test padding is correct for c-order
    float val;
    long long idx;
    long long pad_idx;
    for (int i = 0; i<pad_size[0]; ++i) {
        for (int j = 0; j<pad_size[1]; ++j) {
            for (int k = 0; k<pad_size[2]; ++k) {
                idx = cufftutils::convert_idx(i, j, k, size, column_order);
                pad_idx = cufftutils::convert_idx(i, j, k, pad_size, column_order);

                //val = host_data_input[pad_idx].x; // get the real component
                //printf("idx=%d (%d, %d, %d): %d | ",idx, i, j, k, (int) val);

                if ((i < filterdimA[0]) && (j < filterdimA[1]) && (k < filterdimA[2])) {
                    ASSERT_EQ(host_data_kernel[pad_idx].x, hostF[idx]);
                } else {
                    ASSERT_EQ(host_data_kernel[pad_idx].x, 0.0f);
                }

                if ((i < size[0]) && (j < size[1]) && (k < size[2]) ) {
                    ASSERT_EQ(host_data_input[pad_idx].x, hostI[idx]);
                } else {
                    ASSERT_EQ(host_data_input[pad_idx].x, 0.0f);
                }
            }
            //printf("\n");
        }
    }
}

TEST_F(ConvnCufftTest, DISABLED_ConvnFullImageTest) {
    int size[3] = {2048, 2048, 141};
    int filterdimA[3] = {5, 5, 5};
    int benchmark = 1;
    bool column_order = false;
    int algo = 1;
    int result = 0;
    int N = size[0] * size[1] * size[2];
    int N_kernel = filterdimA[0] * filterdimA[1] * filterdimA[2];
    
    printf("Initializing cufft sin array\n");
    float* data = new float[N]; 
    float* kernel = new float[N_kernel]; 

    for (int i=0; i < N; i++)
        data[i] = sin(i);

    //printf("Sin array created\n");

    //printf("Initializing kernel\n");
    for (int i=0; i < N_kernel; i++)
        kernel[i] = sin(i);

    printf("Kernel created\n");

    printf("Testing convolution\n");
    result = cufftutils::conv_handler(data, kernel, data, algo, size,
            filterdimA, column_order, benchmark);

}

TEST_F(ConvnCufftTest, ConvnColumnOrderingTest) {

    // generate params
    int algo = 0;
    bool column_order = false;
    int benchmark = 0;
    int size[] = {50, 50, 50};
    int filterdimA[] = {5, 5, 5};
    int filtersize = filterdimA[0]*filterdimA[1]*filterdimA[2];
    int insize = size[0]*size[1]*size[2];

    // Create a random filter and image
    float* hostI;
    float* hostF;
    float* hostI_column;
    float* hostI_reverted;
    float* hostF_column;
    float* hostO;
    float* hostO_column;

    hostI = new float[insize];
    hostF = new float[filtersize];
    hostI_column = new float[insize];
    hostI_reverted = new float[insize];
    hostF_column = new float[filtersize];
    hostO = new float[insize];
    hostO_column = new float[insize];

    // Create two random images
    initImage(hostI, insize);
    initImage(hostF, filtersize);

    printf("Matrix conversions\n");
    cufftutils::convert_matrix(hostI, hostI_column, size, column_order);
    cufftutils::convert_matrix(hostF, hostF_column, filterdimA, column_order);
    cufftutils::convert_matrix(hostI_column, hostI_reverted, size, !column_order);

    //convert back to original then check the two matrices
    printf("Check double matrix conversion is equal\n");
    long long idx;
    long long col_idx;
    for (int i = 0; i<size[0]; ++i) {
        for (int j = 0; j<size[1]; ++j) {
            for (int k = 0; k<size[2]; ++k) {
                idx = cufftutils::convert_idx(i, j, k, size, column_order);
                //val = host_data_input[pad_idx].x; // get the real component
                //printf("idx=%d (%d, %d, %d): %d | ",idx, i, j, k, (int) val);
                ASSERT_EQ(hostI[idx], hostI_reverted[idx]);
            }
            //printf("\n");
        }
    }

    printf("original order column_order:%d\n", column_order);
    cufftutils::conv_handler(hostI, hostF, hostO, algo, size, filterdimA, column_order, benchmark);
    printf("switch order column_order\n", !column_order);
    cufftutils::conv_handler(hostI_column, hostF_column, hostO_column, algo, size, filterdimA, !column_order, benchmark);

    //convert back to original then check the two matrices
    for (int i = 0; i<size[0]; ++i) {
        for (int j = 0; j<size[1]; ++j) {
            for (int k = 0; k<size[2]; ++k) {
                idx = cufftutils::convert_idx(i, j, k, size, column_order);
                col_idx = cufftutils::convert_idx(i, j, k, size, !column_order);
                //val = host_data_input[pad_idx].x; // get the real component
                //printf("idx=%d (%d, %d, %d): %d | ",idx, i, j, k, (int) val);
                EXPECT_NEAR(hostO[idx], hostO_column[col_idx], 1.0);
            }
            //printf("\n");
        }
    }


}

//TEST_F(ConvnCufftTest, ConvnBasicTest) {
    //// process args
    //const int channels = 500;
    //const int height = 500;
    //const int width = 100;
    //const int image_bytes = channels * height * width * sizeof(float);
    //std::cout << "Creating image" << std::endl;
    //float image[channels][height][width];
    //// TODO change this to a calloc
    //for (int channel = 0; channel < channels; ++channel) {
        //for (int row = 0; row < height; ++row) {
            //for (int col = 0; col < width; ++col) {
                //// image[batch][channel][row][col] = std::rand();
                //image[batch][channel][row][col] = 0.0f;
            //}
        //}
    //}
    //std::cout << "Image created" << std::endl;

    //const int kernel_channels = 5;
    //const int kernel_height = 5;
    //const int kernel_width = 5;
    //float kernel[kernel_height][kernel_width][kernel_channels];
    //for (int channel = 0; channel < kernel_channels; ++channel) {
        //for (int row = 0; row < kernel_height; ++row) {
            //for (int col = 0; col < kernel_width; ++col) {
                //// image[batch][channel][row][col] = std::rand();
                //kernel[channel][row][col] = 0.0f;
            //}
        //}
    //}

    //std::cout << "Kernel Created" << std::endl;

    //float* h_output;
    //cufftutils::conv_handler(image, hostF, hostO, algo, dimA, filterdimA, column_order, benchmark);
    //h_output = cufftutils::convn((float *) image, channels, height, width, (float *) kernel, kernel_channels, kernel_height, kernel_width);
   
    //for (int channel = 0; channel < channels; ++channel) {
        //for (int row = 0; row < height; ++row) {
            //for (int col = 0; col < width; ++col) {
                //// image[batch][channel][row][col] = std::rand();
                //ASSERT_EQ(image[channel][row][col], 0.0f);
            //}
        //}
    //}
//}

} // namespace
