#include "spdlog/spdlog.h"

#include "gpudevice.h"

namespace cudautils {

int get_gpu_num() {
    int num_gpus;

    cudaGetDeviceCount(&num_gpus);
    if (num_gpus < 1) {
        auto logger = spdlog::get("mex_logger");
        logger->error("cudautils::get_gpu_num cannot get # of gpus");
        return -1;
    }

    return num_gpus;
}

}

