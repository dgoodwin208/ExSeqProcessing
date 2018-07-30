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

#include "sift.h"
#include "mexutil.h"
#include "sift_bridge.h"
/*#include "sift_types.h"*/
#include "gpudevice.h"

#include "cuda_task_executor.h"

#include "spdlog/spdlog.h"
#include "stdlib.h"

int main(int argc, char* argv[]) {

    /*if (argc < 4) {*/
        /*std::cout << "Usage: " << argv[0] << " [in image file] [in mask map file] [out interpolated image file]" << std::endl;*/
        /*return 1;*/
    /*}*/

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

        /*std::string in_image_filename1(argv[1]);*/
        /*std::string in_map_filename2  (argv[2]);*/
        /*std::string in_image_filename1("img_2kypts.bin");*/
        std::string in_image_filename1("image_ones.bin");
        std::string in_map_filename2  ("map_2kypts.bin");
        unsigned int x_size, y_size, z_size, x_size1, y_size1, z_size1;
        /*x_size = atoi(argv[4]);*/
        /*y_size = atoi(argv[5]);*/
        /*z_size = atoi(argv[6]);*/

        int keypoint_num;
        try {
            keypoint_num = atoi(argv[1]);
        } catch (const spdlog::spdlog_ex& ex) {
            std::cout << "Must provide # of keypoints `$./sift_cuda 1` " << ex.what() << std::endl;
            return 1;
        }

        logger->info("# of keypoints = {}", keypoint_num);
        x_size = 2048;
        y_size = 2048;
        z_size = 251;
        x_size1 = x_size;
        y_size1 = y_size;
        z_size1 = z_size;

        /*unsigned int x_size, y_size, z_size, x_size1, y_size1, z_size1;*/
        std::ifstream fin1(in_image_filename1, std::ios::binary);
        /*fin1.read((char*)&x_size, sizeof(unsigned int));*/
        /*fin1.read((char*)&y_size, sizeof(unsigned int));*/
        /*fin1.read((char*)&z_size, sizeof(unsigned int));*/

        std::ifstream fin2(in_map_filename2, std::ios::binary);
        /*fin2.read((char*)&x_size1, sizeof(unsigned int));*/
        /*fin2.read((char*)&y_size1, sizeof(unsigned int));*/
        /*fin2.read((char*)&z_size1, sizeof(unsigned int));*/

        if (x_size != x_size1 || y_size != y_size1 || z_size != z_size1) {
            logger->error("the dimension of image and map is not the same. image({},{},{}), map({},{},{})",
                    x_size, y_size, z_size, x_size1, y_size1, z_size1);
            fin1.close();
            fin2.close();
            return 1;
        }

        // create image
        long image_size = x_size * y_size * z_size;
        double* in_image = (double*) malloc(image_size * sizeof(double));
        int8_t* in_map = (int8_t*) malloc(image_size * sizeof(int8_t));
        for (long i=0; i < image_size; i++) {
            in_image[i] = rand() % 100 + 1.0;
            in_map[i] = 1.0;
        }

        // create map
        long long idx;
        for (int i=0; i < keypoint_num; i++) {
            // warning not evenly distributed across the image
            idx = (x_size * rand()) % image_size;
            in_map[idx] = 0.0; // select this point for processing
        }

        /*fin2.read((char*)in_map  .data(), image_size * sizeof(int8_t));*/
        fin1.close();
        fin2.close();

        const unsigned int num_streams = 20;
        int num_gpus = cudautils::get_gpu_num();
        logger->info("# of gpus = {}", num_gpus);
        logger->info("# of streams = {}", num_streams);
        logger->info("# of keypoints = {}", keypoint_num);

        /*std::vector<double> out_interp_image(x_size * y_size * z_size);*/

        const unsigned int x_sub_size = min(2048, x_size);
        const unsigned int y_sub_size = min(2048, y_size / num_gpus);
        const unsigned int dx = min(256, x_sub_size);
        const unsigned int dy = min(256, y_sub_size);
        const unsigned int dw = 0;

        cudautils::SiftParams sift_params;
        double* fv_centers = sift_defaults(&sift_params,
                x_size, y_size, z_size, keypoint_num);

        logger->info("x_size={},y_size={},z_size={},x_sub_size={},y_sub_size={},dx={},dy={},dw={}",
                x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw);

        try {
            cudautils::Keypoint_store keystore;

            cudautils::sift_bridge(
                    logger, x_size, y_size, z_size, x_sub_size, y_sub_size, dx,
                    dy, dw, num_gpus, num_streams, in_image, in_map,
                    sift_params, fv_centers, &keystore);

            /*std::shared_ptr<cudautils::Sift> ni =*/
                /*std::make_shared<cudautils::Sift>(x_size, y_size, z_size,*/
                        /*x_sub_size, y_sub_size, dx, dy, dw, num_gpus,*/
                        /*num_streams, sift_params, fv_centers);*/

            /*cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);*/

            /*logger->info("setImage start");*/
            /*ni->setImage(in_image);*/
            /*logger->info("setImage end");*/
            /*ni->setMapToBeInterpolated(in_map);*/
            /*logger->info("setMap end");*/

            /*logger->info("calc start");*/
            /*executor.run();*/
            /*logger->info("calc end");*/

            /*logger->info("getKeystore start");*/
            /*ni->getKeystore(&keystore);*/
            /*logger->info("getKeystore end");*/

            /*mxArray* mxKeystore;*/
            /*// Convert the output keypoints*/
            /*if ((mxKeystore = kp2mx(&keystore, sift_params)) == NULL)*/
                /*logger->error("keystore to mex error occurred");*/

            free(in_image);
            free(in_map);

        } catch (...) {
            logger->error("internal unknown error occurred");
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

