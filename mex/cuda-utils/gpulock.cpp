#include <string>
#include <fstream>
#include <chrono>
#include <dirent.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/syscall.h>

#include "gpulock.h"

#include "spdlog/spdlog.h"


namespace cudautils {

GPULock::GPULock(const int num_gpus)
    : num_gpus_(num_gpus),
      lock_dir_("/tmp/.exseqproc"),
      locked_gpu_id_(-1) {

    struct stat s;
    if (stat("logs", &s) < 0) {
        mkdir("logs", 0755);
    }

    logger_ = spdlog::get("mex_logger");
    if (logger_ == nullptr) {
        logger_ = spdlog::basic_logger_mt("mex_logger", "logs/mex.log");
    }

    if (stat(lock_dir_.c_str(), &s) < 0) {
        logger_->warn("not exist a directory for gpu lock.");
        if (mkdir(lock_dir_.c_str(), 0777) == 0) {
            logger_->info("created {}", lock_dir_);
        }
    }

    std::string all_lock_filename = lock_dir_ + "/all_gpus.lock";
    all_filelock_ = std::make_shared<FileLock>(all_lock_filename);
    for (size_t i = 0; i < num_gpus_; i++) {
        std::string lock_filename = lock_dir_ + "/gpu" + std::to_string(i) + ".lock";
        filelocks_.push_back(std::make_shared<FileLock>(lock_filename));
    }
}

int GPULock::trylock(const int timeout_sec) {
    if (locked_gpu_id_ != -1) {
        logger_->warn("already locked: gpu_id={}", locked_gpu_id_);
        return locked_gpu_id_;
    }

    mode_t old_umask = umask(0);

    auto max_sec = std::chrono::seconds(timeout_sec);
    auto sum_msec = std::chrono::milliseconds(0);
    auto interval_msec = std::chrono::milliseconds(100);

    // get all-gpus lock
    while (1) {
        if (all_filelock_->trylock() == 0) {
            break;
        }

        std::this_thread::sleep_for(interval_msec);
        sum_msec += interval_msec;

        if (timeout_sec >= 0 && sum_msec >= max_sec) {
            logger_->warn("timeout of all-gpus lock.. ({} sec)", timeout_sec);
            umask(old_umask);
            return -1;
        }
    }

    // get one gpu lock
    sum_msec = std::chrono::milliseconds(0);
    while (1) {
        for (size_t i = 0; i < num_gpus_; i++) {
            if (filelocks_[i]->trylock() == 0) {
                locked_gpu_id_ = i;
                break;
            }
        }
        if (locked_gpu_id_ != -1) {
            break;
        }

        std::this_thread::sleep_for(interval_msec);
        sum_msec += interval_msec;

        if (timeout_sec >= 0 && sum_msec >= max_sec) {
            logger_->warn("timeout of one gpu lock.. ({} sec)", timeout_sec);
            all_filelock_->unlock();
            umask(old_umask);
            return -1;
        }
    }

    // release all-gpus lock
    all_filelock_->unlock();

    umask(old_umask);

    return locked_gpu_id_;
}

int GPULock::trylockall(const int timeout_sec) {

    mode_t old_umask = umask(0);

    auto max_sec = std::chrono::seconds(timeout_sec);
    auto sum_msec = std::chrono::milliseconds(0);
    auto interval_msec = std::chrono::milliseconds(100);

    // get all-gpus lock
    while (1) {
        if (all_filelock_->trylock() == 0) {
            break;
        }

        std::this_thread::sleep_for(interval_msec);
        sum_msec += interval_msec;

        if (timeout_sec >= 0 && sum_msec >= max_sec) {
            logger_->warn("timeout of all-gpus lock.. ({} sec)", timeout_sec);
            umask(old_umask);
            return -1;
        }
    }

    // get all each-gpu lock
    sum_msec = std::chrono::milliseconds(0);
    int lock_count = 0;
    while (1) {
        for (size_t i = 0; i < num_gpus_; i++) {
            if (! filelocks_[i]->isLocked() && filelocks_[i]->trylock() == 0) {
                lock_count++;
            }
        }
        if (lock_count == num_gpus_) {
            break;
        }

        std::this_thread::sleep_for(interval_msec);
        sum_msec += interval_msec;

        if (timeout_sec >= 0 && sum_msec >= max_sec) {
            logger_->warn("timeout of all each-gpu lock.. ({} sec)", timeout_sec);
            for (size_t i = 0; i < num_gpus_; i++) {
                filelocks_[i]->unlock();
            }
            all_filelock_->unlock();
            umask(old_umask);
            return -1;
        }
    }

    umask(old_umask);

    return 0;
}

int GPULock::unlock() {
    if (locked_gpu_id_ == -1) {
        logger_->error("not lock any gpu");
        return -1;
    }

    if (filelocks_[locked_gpu_id_]->unlock() != 0) {
        logger_->error("failed to unlock: gpu_id={}", locked_gpu_id_);
        return -1;
    }

    locked_gpu_id_ = -1;

    return 0;
}

int GPULock::unlockall() {
    if (! all_filelock_->isLocked()) {
        logger_->error("not lock all-gpus");
        return -1;
    }

    for (size_t i = 0; i < num_gpus_; i++) {
        filelocks_[i]->unlock();
    }
    all_filelock_->unlock();

    return 0;
}

}

