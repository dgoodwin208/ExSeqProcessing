#include "gtest/gtest.h"

#include "cufftutils.h"
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

TEST_F(ConvnCufftTest, InitializePadTest) {
    int size[3] = {3, 3, 4};
    int filterdimA[3] = {4, 2, 2};
    int benchmark = 1;
    bool column_order = 1;
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
    for (int i = 0; i<pad_size[0]; ++i) {
        for (int j = 0; j<pad_size[1]; ++j) {
            for (int k = 0; k<pad_size[2]; ++k) {
                //idx = k + j*size[2] + size[2]*size[1]*i;
                idx = cufftutils::convert_idx(i, j, k, pad_size, column_order);
                val = hostI[idx];
                printf("idx=%d (%d, %d, %d): %d | ",idx, i, j, k, (int) val);
            }
        }
    }
}

TEST_F(ConvnCufftTest, DISABLED_ConvnFullImageTest) {
    int size[3] = {2048, 2048, 141};
    int filterdimA[3] = {5, 5, 5};
    int benchmark = 1;
    bool column_order = 1;
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

TEST_F(ConvnCufftTest, DISABLED_ConvnBasicTest) {

    // generate params
    int algo = 0;
    bool column_order = 1;
    int benchmark = 0;
    int dimA[] = {50, 50, 10};
    int filterdimA[] = {5, 5, 5};
    int filtersize = filterdimA[0]*filterdimA[1]*filterdimA[2];
    int insize = dimA[0]*dimA[1]*dimA[2];
    int outsize = insize;

    // Create a random filter and image
    float* hostI;
    float* hostF;
    float* hostO;

    hostI = (float*)calloc (insize, sizeof(hostI[0]) );
    hostF = (float*)calloc (filtersize, sizeof(hostF[0]) );
    hostO = (float*)calloc (outsize, sizeof(hostO[0]) );

    // Create two random images
    initImage(hostI, insize);
    initImage(hostF, filtersize);

    cufftutils::conv_handler(hostI, hostF, hostO, algo, dimA, filterdimA, column_order, benchmark);

    //TODO test the convolution output hostO
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
