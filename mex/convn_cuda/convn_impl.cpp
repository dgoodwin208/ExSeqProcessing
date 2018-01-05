#include <sstream>
#include <iomanip>
#include <exception>
#include <chrono>
#include <cassert>
#include <cstring>
#include <semaphore.h>
#include <fcntl.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include "convn_impl.h"
#include "cuda-utils/convn_cu.h"

void ConvnImpl::run(float **image, float **kernel, float **h_output) {
    convn_cuda(image, kernel, h_output);
}


void ConvnImpl::waitForTasks(const std::string& task_name, std::vector<std::future<int>>& futures) {
    logger_->info("[{}] {} waiting...", basename_, task_name);
    for (size_t i = 0; i < futures.size(); i++) {
        int ret = futures[i].get();
        if (ret == -1) {
            logger_->error("[{}] {} [{}] failed - {}", basename_, task_name, i, ret);
            throw ExceptionToMATLAB("MATLAB:quantilenorm_impl:failTasks", task_name + "failed.");
        }
    }
    logger_->info("[{}] {} done", basename_, task_name);
}

