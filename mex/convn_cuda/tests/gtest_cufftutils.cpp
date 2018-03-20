#include "gtest/gtest.h"

#include "cufftutils.h"

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

TEST_F(ConvnCufftTest, FFTBasicTest) {
    int size[3] = {204, 204, 50};
    int filterdimA[3] = {5, 5, 5};
    int benchmark = 1;
    int pad = 1;
    int algo = 1;
    int column_order = 0;
    int N = size[0] * size[1] * size[2];
    int N_kernel = filterdimA[0] * filterdimA[1] * filterdimA[2];
    
    printf("Initializing cufft sin array\n");
    float* data = new float[N]; 
    float* outArray = new float[N]; 
    float* kernel = new float[N_kernel]; 

    for (int i=0; i < N; i++)
        data[i] = sin(i);

    printf("Sin array created\n");

    printf("Initializing kernel\n");
    for (int i=0; i < N_kernel; i++)
        kernel[i] = sin(i);

    printf("Kernel created\n");

    printf("Testing fft\n");
    cufftutils::fft3(data, size, size, outArray, column_order);
}

TEST_F(ConvnCufftTest, ConvnOriginalTest) {
    int size[3] = {204, 204, 50};
    int filterdimA[3] = {5, 5, 5};
    int benchmark = 1;
    int pad = 1;
    int algo = 1;
    int result = 0;
    int N = size[0] * size[1] * size[2];
    int N_kernel = filterdimA[0] * filterdimA[1] * filterdimA[2];
    
    printf("Initializing cufft sin array\n");
    float* data = new float[N]; 
    float* kernel = new float[N_kernel]; 

    for (int i=0; i < N; i++)
        data[i] = sin(i);

    printf("Sin array created\n");

    printf("Initializing kernel\n");
    for (int i=0; i < N_kernel; i++)
        kernel[i] = sin(i);

    printf("Kernel created\n");

    printf("Testing convolution\n");
    result = cufftutils::conv_handler(data, kernel, data, algo, size,
            filterdimA, pad, benchmark);

}

TEST_F(ConvnCufftTest, ConvnSampleTest) {

    // generate params
    int algo = 0;
    int pad = 1;
    int benchmark = 0;
    int dimA[] = {500, 500, 100};
    int filterdimA[] = {5, 5, 5};
    int filtersize = filterdimA[0]*filterdimA[1]*filterdimA[2];
    int insize = dimA[0]*dimA[1]*dimA[2];
    int outdimA[3] = {500, 500, 100};
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

    cufftutils::conv_handler(hostI, hostF, hostO, algo, dimA, filterdimA, pad, benchmark);

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
    //cufftutils::conv_handler(image, hostF, hostO, algo, dimA, filterdimA, pad, benchmark);
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
