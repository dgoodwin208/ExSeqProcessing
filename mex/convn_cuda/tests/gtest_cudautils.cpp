#include "gtest/gtest.h"

#include "convn.h"

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

    //std::vector<uint16_t> keys;
    //std::vector<unsigned int> values;

    //const size_t DATA_SIZE = 10000;

    //std::mt19937 mt(1);
    //for (size_t i = 0; i < DATA_SIZE; i++) {
        //keys.push_back(mt());
        //values.push_back(i);
    //}
    //std::vector<uint16_t> keys2(keys.begin(), keys.end());

    //cudautils::radixsort(keys2, values);

    //for (size_t i = 0, j = 1; i < DATA_SIZE - 1; i++, j++) {
        //ASSERT_LE(keys2[i], keys2[j]);
    //}

    //std::vector<double> keys3;
    //for (size_t i = 0; i < DATA_SIZE; i++) {
        //keys3.push_back((double)keys2[i]);
    //}

    //cudautils::radixsort(values, keys3);

    //for (size_t i = 0, j = 1; i < DATA_SIZE - 1; i++, j++) {
        //ASSERT_LE(values[i], values[j]);
    //}
    //for (size_t i = 0; i < DATA_SIZE; i++) {
        //ASSERT_EQ((double)keys[i], keys3[i]);
    //}
