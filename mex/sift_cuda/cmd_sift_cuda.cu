/*=================================================================
 * cmd_sift_cuda.cu - perform sift on volume image data 
 *
 *  sift_cuda(vol_image, map)
 *
 *  Input:
 *    vol_image:  volume image data
 *    map:        mask map data (1: mask, 0: hole)
 *
 *  Output:
 *    ?
 *
 *=================================================================*/
 

#include <fstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <stdexcept>

#include "sift.h"
#include "mexutil.h"
#include "sift_bridge.h"
#include "gpudevice.h"

#include "cuda_task_executor.h"

#include "spdlog/spdlog.h"
#include "stdlib.h"

int main(int argc, char* argv[]) {

    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " [in image file] [in mask map file] [out interpolated image file]" << std::endl;
        return 1;
    }

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
        logger->info("{:=>50}", " sift_cuda start");

        std::string in_image_filename1(argv[1]);
        std::string in_map_filename2  (argv[2]);
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

        std::ifstream fin2(in_map_filename2, std::ios::binary);
        if (fin2.is_open()) {
            fin2.read((char*)&x_size1, sizeof(unsigned int));
            fin2.read((char*)&y_size1, sizeof(unsigned int));
            fin2.read((char*)&z_size1, sizeof(unsigned int));
        } else { 
            throw std::invalid_argument( "Unable to open or find file: `test_map.bin` in current directory");
        }

        if (x_size != x_size1 || y_size != y_size1 || z_size != z_size1) {
            logger->error("the dimension of image and map is not the same. image({},{},{}), map({},{},{})",
                    x_size, y_size, z_size, x_size1, y_size1, z_size1);
            fin1.close();
            fin2.close();
            throw std::invalid_argument("Dimension of image and map is not the same");
        }

        std::vector<double> in_image(x_size * y_size * z_size);
        std::vector<int8_t> in_map  (x_size * y_size * z_size);
        fin1.read((char*)in_image.data(), x_size * y_size * z_size * sizeof(double));
        fin2.read((char*)in_map  .data(), x_size * y_size * z_size * sizeof(int8_t));
        fin1.close();
        fin2.close();

        cudautils::SiftParams sift_params;
        double* fv_centers = sift_defaults(&sift_params,
                x_size, y_size, z_size, 0);
        int stream_num = 20;
        int x_substream_stride = 256;
        int y_substream_stride = 256;
        
        int num_gpus = cudautils::get_gpu_num();
        logger->info("# of gpus = {}", num_gpus);
        logger->info("# of streams = {}", stream_num);

        const unsigned int x_sub_size = x_size;
        const unsigned int y_sub_size = y_size / num_gpus;
        const unsigned int dw = 0;

        logger->info("x_size={},y_size={},z_size={},x_sub_size={},y_sub_size={},x_substream_stride={},y_substream_stride={},dw={},# of streams={}",
                x_size, y_size, z_size, x_sub_size, y_sub_size,
                x_substream_stride, y_substream_stride, dw, stream_num);

        cudautils::Keypoint_store keystore;
        try {

            cudautils::sift_bridge( logger, x_size, y_size, z_size, x_sub_size,
                    y_sub_size, x_substream_stride, y_substream_stride, dw,
                    num_gpus, stream_num, &in_image[0], &in_map[0],
                    sift_params, fv_centers, &keystore);

        } catch (...) {
            logger->error("Internal unknown error occurred during CUDA execution");
            throw;
        }

        logger->info("save Keystore start");
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
            throw std::invalid_argument( "Unable to open output file");
        }

        logger->info("{:=>50}", " sift_cuda end");

        logger->flush();
        spdlog::drop_all();
    } catch (...) {
        logger->flush();
        throw;
    }

    return 0;
}

