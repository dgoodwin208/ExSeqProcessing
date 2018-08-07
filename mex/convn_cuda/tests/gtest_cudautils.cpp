#include "gtest/gtest.h"

#include "convn.h"
#include "cudnnutils.h"

#include <vector>
#include <cstdint>
#include <random>

namespace {

class ConvnTest : public ::testing::Test {
protected:
    ConvnTest() {
    }
    virtual ~ConvnTest() {
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

static void generateStrides(const int* dimA, int* strideA, int nbDims, bool isNchw) {
    if (isNchw) {
        strideA[nbDims-1] = 1 ;
        for(int d = nbDims-2 ; d >= 0 ; d--) {
            strideA[d] = strideA[d+1] * dimA[d+1] ;
        }
    } else {
        strideA[1] = 1;
        strideA[nbDims-1] = strideA[1]*dimA[1];
        for(int d = nbDims-2 ; d >= 2 ; d--) {
            strideA[d] = strideA[d+1] * dimA[d+1] ;
        }
        strideA[0] = strideA[2]*dimA[2];
    }
}

TEST_F(ConvnTest, ConvnSampleTest) {

    // generate params
    int algo = 0;
    int benchmark = 0;
    int dimA[] = {1, 8, 32, 32};
    int filterdimA[] = {8, 8, 3, 3};
    int filtersize = filterdimA[0]*filterdimA[1]*filterdimA[2]*filterdimA[3];

    int strideA[] = {8192, 1024, 32, 1};
    int nbDims = 4;
    bool isNchw = true;
    generateStrides(dimA, strideA, nbDims, isNchw);
    int insize = strideA[0]*dimA[0];
    
    int outdimA[] = {1, 8, 30, 30};
    outdimA[0] = dimA[0];
    int outstrideA[] = {7200, 900, 30, 1};
    int outsize = outstrideA[0]*outdimA[0];

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

    cudnnutils::conv_handler(hostI, hostF, hostO, algo, dimA, filterdimA, benchmark);

    //TODO test the convolution output hostO
}

TEST_F(ConvnTest, ConvnBasicTest) {
    // process args
    const int batch_size = 1;
    const int channels = 3;
    const int height = 300;
    const int width = 300;
    const int image_bytes = batch_size * channels * height * width * sizeof(float);
    std::cout << "Creating image" << std::endl;
    float image[batch_size][channels][height][width];
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int channel = 0; channel < channels; ++channel) {
            for (int row = 0; row < height; ++row) {
                for (int col = 0; col < width; ++col) {
                    // image[batch][channel][row][col] = std::rand();
                    image[batch][channel][row][col] = 0.0f;
                }
            }
        }
    }
    std::cout << "Image created" << std::endl;

    const int kernel_channels = 3;
    const int kernel_height = 3;
    const int kernel_width = 3;
    float kernel[kernel_height][kernel_width][kernel_channels];
    for (int channel = 0; channel < kernel_channels; ++channel) {
        for (int row = 0; row < kernel_height; ++row) {
            for (int col = 0; col < kernel_width; ++col) {
                // image[batch][channel][row][col] = std::rand();
                kernel[channel][row][col] = 0.0f;
            }
        }
    }
    // toy kernel 
    // clang-format off
    //float kernel[kernel_height][kernel_width] = {
        //{1, 1, 1},
        //{1, -8, 1},
        //{1, 1, 1}
    //};
    // clang-format on
    std::cout << "Kernel Created" << std::endl;


    //// copy into higher dim, once for each channel, once for each output feature map (out_channel)
    //float h_kernel[batch_size][channels][kernel_height][kernel_width];
    //for (int out_channel = 0; out_channel < out_channels; ++out_channel) {
       //for (int channel = 0; channel < channels; ++channel) {
           //for (int row = 0; row < kernel_height; ++row) {
               //for (int column = 0; column < kernel_width; ++column) {
                   //h_kernel[out_channel][channel][row][column] = kernel[row][column];
               //}
           //}
       //}
    //}

    float* h_output;
    h_output = cudautils::convn((float *) image, channels, height, width, (float *) kernel, kernel_channels, kernel_height, kernel_width);
   
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int channel = 0; channel < channels; ++channel) {
            for (int row = 0; row < height; ++row) {
                for (int col = 0; col < width; ++col) {
                    // image[batch][channel][row][col] = std::rand();
                    ASSERT_EQ(image[batch][channel][row][col], 0.0f);
                }
            }
        }
    }
}

} // namespace
