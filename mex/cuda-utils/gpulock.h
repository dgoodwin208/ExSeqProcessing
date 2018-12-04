#ifndef __GPULOCK_H__
#define __GPULOCK_H__

#include <vector>
#include <string>

#include "filelock.h"
#include "spdlog/spdlog.h"

namespace cudautils {

class GPULock {
    const int num_gpus_;
    const std::string lock_dir_;

    int locked_gpu_id_;

    std::shared_ptr<FileLock> all_filelock_;
    std::vector<std::shared_ptr<FileLock>> filelocks_;

    std::shared_ptr<spdlog::logger> logger_;

public:
    GPULock(const int num_gpus);
    int trylock(const int timeout_sec = 0);
    int trylockall(const int timeout_sec = 0);
    int unlock();
    int unlockall();
};

}

#endif

