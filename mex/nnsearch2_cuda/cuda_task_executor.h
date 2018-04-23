#ifndef __CUDA_TASK_EXECUTOR_H__
#define __CUDA_TASK_EXECUTOR_H__

#include <vector>
#include <string>
#include <memory>

#include "cuda_task.h"

#include "spdlog/spdlog.h"

namespace parallelutils {

#define DEBUG_NO_THREADING

class CudaTaskExecutor {
    int num_gpus_;
    int num_streams_;

    std::shared_ptr<CudaTask> task_;

    std::shared_ptr<spdlog::logger> logger_;

public:
    CudaTaskExecutor(const int num_gpus, const int num_streams, std::shared_ptr<CudaTask> task);

    int run();
    int runOnGPU(const int gpu_id);
    int runOnStream(const int gpu_id, const int stream_id, const int task_id);

};

}

#endif

