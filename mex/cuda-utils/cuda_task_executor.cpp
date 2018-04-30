#include <sstream>
#include <thread>
#include <future>

#include "cuda_task_executor.h"

#include "spdlog/spdlog.h"


namespace cudautils {

CudaTaskExecutor::CudaTaskExecutor(
        const int num_gpus,
        const int num_streams,
        std::shared_ptr<CudaTask> task)
    : num_gpus_(num_gpus), num_streams_(num_streams), task_(task) {
    logger_ = spdlog::get("console");
    if (! logger_) {
        logger_ = spdlog::stdout_logger_mt("console");
    }
}

int CudaTaskExecutor::run() {
    logger_->debug("===== run: num_gpus={} num_streams={}", num_gpus_, num_streams_);

    try {
        task_->prerun();

#ifndef DEBUG_NO_THREADING
        std::launch policy = std::launch::async;
#else
        std::launch policy = std::launch::deferred;
#endif
        std::vector<std::future<int>> futures;
        for (size_t gpu_id = 0; gpu_id < num_gpus_; gpu_id++) {
            futures.push_back(std::async(policy, &CudaTaskExecutor::runOnGPU, this, gpu_id));
        }

        for (size_t i = 0; i < futures.size(); i++) {
            int ret = futures[i].get();
            if (ret != 1) {
                logger_->error("gpu_id[{}] has failed in thread.", i);
                throw std::string("run() thread faild."); //TODO
            }
        }

        task_->postrun();

    } catch (...) {//TODO
        logger_->error("run() has failed in thread.");
        return 0;
    }

    return 1;
}

int CudaTaskExecutor::runOnGPU(const int gpu_id) {
    logger_->debug("===== runOnGPU: gpu_id={}", gpu_id);

    try {
        int num_gpu_tasks = task_->getNumOfGPUTasks(gpu_id);
        for (int gpu_task_id = 0; gpu_task_id < num_gpu_tasks; gpu_task_id++) {
            task_->runOnGPU(gpu_id, gpu_task_id);

#ifndef DEBUG_NO_THREADING
            std::launch policy = std::launch::async;
#else
            std::launch policy = std::launch::deferred;
#endif
            std::vector<std::future<int>> futures;
            for (int stream_id = 0; stream_id < num_streams_; stream_id++) {
                futures.push_back(std::async(policy, &CudaTaskExecutor::runOnStream, this,
                            gpu_id, stream_id, gpu_task_id));
            }

            for (size_t i = 0; i < futures.size(); i++) {
                int ret = futures[i].get();
                if (ret != 1) {
                    logger_->error("stream_id[{}] has failed in thread.", i);
                    throw std::string("runOnGPU() thread faild."); //TODO
                }
            }

            task_->postrunOnGPU(gpu_id, gpu_task_id);
        }

    } catch (...) {//TODO
        logger_->error("runOnGPU() has failed in thread.");
        return 0;
    }

    return 1;
}

int CudaTaskExecutor::runOnStream(const int gpu_id, const int stream_id, const int gpu_task_id) {
    logger_->debug("===== runOnStream: gpu_id={}, stream_id={}, gpu_task_id={}",
            gpu_id, stream_id, gpu_task_id);

    try {
        task_->runOnStream(gpu_id, stream_id, gpu_task_id);
    } catch (...) {//TODO
        logger_->error("runOnStream() has failed in thread.");
        return 0;
    }

    return 1;
}

}

