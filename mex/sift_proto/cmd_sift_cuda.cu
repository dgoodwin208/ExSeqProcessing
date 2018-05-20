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

#include "sift.h"
#include "gpudevice.h"

#include "cuda_task_executor.h"

#include "spdlog/spdlog.h"


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
        std::string out_interp_image_filename(argv[3]);

        unsigned int x_size, y_size, z_size, x_size1, y_size1, z_size1;
        std::ifstream fin1(in_image_filename1, std::ios::binary);
        fin1.read((char*)&x_size, sizeof(unsigned int));
        fin1.read((char*)&y_size, sizeof(unsigned int));
        fin1.read((char*)&z_size, sizeof(unsigned int));

        std::ifstream fin2(in_map_filename2, std::ios::binary);
        fin2.read((char*)&x_size1, sizeof(unsigned int));
        fin2.read((char*)&y_size1, sizeof(unsigned int));
        fin2.read((char*)&z_size1, sizeof(unsigned int));

        if (x_size != x_size1 || y_size != y_size1 || z_size != z_size1) {
            logger->error("the dimension of image and map is not the same. image({},{},{}), map({},{},{})",
                    x_size, y_size, z_size, x_size1, y_size1, z_size1);
            fin1.close();
            fin2.close();
            return 1;
        }

        std::vector<double> in_image(x_size * y_size * z_size);
        std::vector<int8_t> in_map  (x_size * y_size * z_size);
        fin1.read((char*)in_image.data(), x_size * y_size * z_size * sizeof(double));
        fin2.read((char*)in_map  .data(), x_size * y_size * z_size * sizeof(int8_t));
        fin1.close();
        fin2.close();

//        int num_gpus = 1;
        int num_gpus = cudautils::get_gpu_num();
        logger->info("# of gpus = {}", num_gpus);

        std::vector<double> out_interp_image(x_size * y_size * z_size);

        const unsigned int x_sub_size = min(2048, x_size);
        const unsigned int y_sub_size = min(1024, y_size);
        const unsigned int dx = min(256, x_sub_size);
        const unsigned int dy = min(256, y_sub_size);
        const unsigned int dw = 2;

        const unsigned int num_streams = 20;
        logger->info("x_size={},y_size={},z_size={},x_sub_size={},y_sub_size={},dx={},dy={},dw={},# of streams={}",
                x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_streams);

        try {
            std::shared_ptr<cudautils::Sift> ni =
                std::make_shared<cudautils::Sift>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

            cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

            logger->info("setImage start");
            ni->setImage(in_image);
            logger->info("setImage end");
            ni->setMapToBeInterpolated(in_map);
            logger->info("setMap end");

            logger->info("calc start");
            executor.run();
            logger->info("calc end");

            logger->info("getImage start");
            ni->getImage(out_interp_image);
            logger->info("getImage end");

            logger->info("saveImage start");
            std::ofstream fout(out_interp_image_filename, std::ios::binary);
            fout.write((char*)&x_size, sizeof(unsigned int));
            fout.write((char*)&y_size, sizeof(unsigned int));
            fout.write((char*)&z_size, sizeof(unsigned int));

            fout.write((char*)out_interp_image.data(), x_size * y_size * z_size * sizeof(double));
            fout.close();
            logger->info("saveImage end");

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

