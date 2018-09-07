/*=================================================================
 * cmd_convn_cuda.cu - perform convolution on volumetric image data 
 *
 *  ./cmd_convn_cuda vol_image.bin filter.bin output.bin
 *
 *  Input:
 *    vol_image.bin:  volume image data (float)
 *    filter.bin:        filter data (float)
 *
 *  Output:
 *    output.bin: convolved output image (float)
 *
 *=================================================================*/
 

#include <fstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <stdexcept>
#include <iostream>

#include "cufft-utils/cufftutils.h"

#include "spdlog/spdlog.h"
#include "stdlib.h"

int main(int argc, char* argv[]) {

    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " [in image file] [in filter file] [out convolved image file]" << std::endl;
        return 1;
    }

    // Create log
    std::shared_ptr<spdlog::logger> logger;
    try {
        spdlog::set_async_mode(4096, spdlog::async_overflow_policy::block_retry, nullptr, std::chrono::seconds(2));
        spdlog::set_level(spdlog::level::trace);
        logger = spdlog::get("mex_logger");
        if (logger == nullptr) {
            logger = spdlog::basic_logger_mt("mex_logger", "logs/mex.log");
        }
        logger->flush_on(spdlog::level::err);
        //logger->flush_on(spdlog::level::info);
    } catch (const spdlog::spdlog_ex& ex) {
        std::cout << "Log initialization failed: " << ex.what() << std::endl;
        return 1;
    }

    try {
        logger->info("{:=>50}", "cmd_convn_cuda start");

        std::string in_image_filename1(argv[1]);
        std::string in_filter_filename2  (argv[2]);
        std::string out_filename(argv[3]);

        unsigned int x_size, y_size, z_size, x_size1, y_size1, z_size1;

        std::ifstream fin1(in_image_filename1, std::ios::binary);
        if (fin1.is_open()) {
            fin1.read((char*)&x_size, sizeof(unsigned int));
            fin1.read((char*)&y_size, sizeof(unsigned int));
            fin1.read((char*)&z_size, sizeof(unsigned int));
        } else { 
            throw std::invalid_argument( "Unable to open or find file: `test_image.bin` in current directory");
        }

        std::ifstream fin2(in_filter_filename2, std::ios::binary);
        if (fin2.is_open()) {
            fin2.read((char*)&x_size1, sizeof(unsigned int));
            fin2.read((char*)&y_size1, sizeof(unsigned int));
            fin2.read((char*)&z_size1, sizeof(unsigned int));
        } else { 
            throw std::invalid_argument( "Unable to open or find file: `test_filter.bin` in current directory");
        }

        if (x_size < x_size1 || y_size < y_size1 || z_size < z_size1) {
            logger->error("The dimensions of filter can not be greater than that of image. image({},{},{}), filter({},{},{})",
                    x_size, y_size, z_size, x_size1, y_size1, z_size1);
            fin1.close();
            fin2.close();
            throw std::invalid_argument("Dimension of filter greater than image");
        }

        std::vector<float> in_image(x_size * y_size * z_size);
        std::vector<float> outArray(x_size * y_size * z_size);
        std::vector<float> in_filter  (x_size1 * y_size1 * z_size1);
        fin1.read((char*)in_image.data(), x_size * y_size * z_size * sizeof(float));
        fin2.read((char*)in_filter  .data(), x_size * y_size * z_size * sizeof(float));
        fin1.close();
        fin2.close();

        logger->info("image({},{},{}), filter({},{},{})", x_size,
                y_size, z_size, x_size1, y_size1, z_size1);

        unsigned int image_size[3] = {x_size, y_size, z_size};
        unsigned int filter_size[3] = {x_size1, y_size1, z_size1};

        try {
            // generate params
            int algo = 0; // forward convolve
            bool column_order = true; // bool for 
            int benchmark = 1;
            // padding the image to m + n -1 or greater per dimension
            // create a pointer to the real data in the input array
            cufftutils::conv_handler(&in_image[0], &in_filter[0],
                    &outArray[0], algo, image_size, filter_size,
                    column_order, benchmark);

            logger->info("`cufftutils::conv_handler` executed successfully");

        } catch (...) {
            logger->error("Internal unknown error occurred during CUDA execution");
        }

        std::ofstream fout(out_filename, std::ios::binary);
        if (fout.is_open()) {
            fout.write((char*) outArray.data(), sizeof(outArray));
            fout.close();
        } else { 
            throw std::invalid_argument( "Unable to open output file");
        }

        logger->info("{:=>50}", " cmd_convn_cuda end");

        logger->flush();
        spdlog::drop_all();
    } catch (...) {
        logger->flush();
        throw;
    }

    return 0;
}

