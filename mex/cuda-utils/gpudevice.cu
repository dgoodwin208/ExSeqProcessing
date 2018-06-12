#include "spdlog/spdlog.h"

#include "gpudevice.h"

namespace cudautils {

int get_gpu_num() {
    int num_gpus;

    cudaGetDeviceCount(&num_gpus);
    if (num_gpus < 1) {
        auto logger = spdlog::get("mex_logger");
        logger->error("cudautils::get_gpu_num cannot get # of gpus");
        cudaDeviceReset();
        return -1;
    }

    cudaDeviceReset();
    return num_gpus;
}

void get_gpu_mem_size(size_t& free_size, size_t& total_size) {
    cudaMemGetInfo(&free_size, &total_size);
    cudaDeviceReset();
}

void resetDevice() {
    cudaDeviceReset();
}

}

