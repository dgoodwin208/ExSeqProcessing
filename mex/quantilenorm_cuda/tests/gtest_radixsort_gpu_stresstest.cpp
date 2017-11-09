#include "gtest/gtest.h"

#include <vector>
#include <cstdint>
#include <random>
#include <thread>
#include <future>
#include <semaphore.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <cuda_runtime_api.h>
#include "radixsort.h"
#include "spdlog/spdlog.h"

#define MAX_DATA_SIZE_FOR_GPU 1024*1024*180
//#define MAX_DATA_SIZE_FOR_GPU 10
#define IMAGE_X 100
#define IMAGE_Y 100
#define IMAGE_Z 10
#define GPU_NUM 2

namespace {

class RadixSortGPUStressTest : public ::testing::Test {
protected:
    std::vector<unsigned int> keys_;
    std::vector<double> values_;
    std::shared_ptr<spdlog::logger> logger_;

    RadixSortGPUStressTest() {
        logger_ = spdlog::basic_logger_mt("mex_logger", "mex.log");

        mode_t old_umask = umask(0);
        for (size_t i = 0; i < GPU_NUM; i++) {
            std::string sem_name = "/g" + std::to_string(i);
            sem_unlink(sem_name.c_str());
            sem_open(sem_name.c_str(), O_CREAT|O_RDWR, 0777, 1);
        }
        umask(old_umask);

        std::mt19937 mt(1);
        for (size_t i = 0; i < MAX_DATA_SIZE_FOR_GPU; i++) {
            keys_.push_back(mt());
            values_.push_back((double)i);
        }
    }
    virtual ~RadixSortGPUStressTest() {
        for (size_t i = 0; i < GPU_NUM; i++) {
            std::string sem_name = "/g" + std::to_string(i);
            sem_unlink(sem_name.c_str());
        }
    }

    int
    selectGPU(const std::string& target) {
        int idx_gpu = -1;
        for (size_t i = 0; i < GPU_NUM; i++) {
            std::string sem_name = "/g" + std::to_string(i);
            sem_t *sem;
            sem = sem_open(sem_name.c_str(), O_RDWR);
            int ret = errno;
            if (sem == SEM_FAILED) {
                logger_->error("[{}] cannot open semaphore of {}", target, sem_name);
                continue;
            }

            ret = sem_trywait(sem);
            if (ret == 0) {
                logger_->trace("[{}] selectGPU {}", target, sem_name);
                idx_gpu = i;
                cudaSetDevice(idx_gpu);
                break;
            }
        }

        return idx_gpu;
    }

    void
    unselectGPU(const std::string& target, const int idx_gpu) {
        std::string sem_name = "/g" + std::to_string(idx_gpu);
        sem_t *sem;
        sem = sem_open(sem_name.c_str(), O_RDWR);
        int ret = errno;
        if (sem == SEM_FAILED) {
            logger_->error("[{}] cannot open semaphore of {}", target, sem_name);
            return;
        }

        cudaDeviceReset();

        logger_->trace("[{}] unselectGPU {}", target, sem_name);
        ret = sem_post(sem);
        if (ret != 0) {
            logger_->error("[{}] cannot post semaphore of {}", target, sem_name);
            return;
        }
    }

public:
    int
    radixSortThread(const std::string& target) {
        logger_->info("[{}] start", target);

        int ret;
        auto interval_sec = std::chrono::seconds(1);
        while (1) {
            int idx_gpu = selectGPU(target);
            if (idx_gpu >= 0) {
                logger_->info("[{}] radixSort2FromData: idx_gpu = {}", target, idx_gpu);

                try {
                    cudautils::radixsort<unsigned int, double>(keys_, values_);
                } catch (std::exception& ex) {
                    logger_->debug("[{}] radixSort2FromData: {}", target, ex.what());
                    cudaError err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        logger_->error("[{}] {}", target, cudaGetErrorString(err));
                    }
                    unselectGPU(target, idx_gpu);
                    std::this_thread::sleep_for(interval_sec);
                    continue;
                } catch (...) {
                    logger_->debug("[{}] radixSort2FromData: unknown error", target);
                    cudaError err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        logger_->error("[{}] {}", target, cudaGetErrorString(err));
                    }
                    unselectGPU(target, idx_gpu);
                    std::this_thread::sleep_for(interval_sec);
                    continue;
                }

                unselectGPU(target, idx_gpu);
                break;
            } else {
                std::this_thread::sleep_for(interval_sec);
            }
        }
        logger_->info("[{}] end", target);

        return 0;
    }
};


TEST_F(RadixSortGPUStressTest, StressTest) {
    const size_t num_parallel = 200;
    std::vector<std::future<int>> radixsort_futures;

    for (size_t i = 0; i < num_parallel; i++) {
        std::string target = "t" + std::to_string(i);
        radixsort_futures.push_back(std::async(std::launch::async, &RadixSortGPUStressTest::radixSortThread, this, target));
    }

    logger_->info("waiting...");
    for (size_t i = 0; i < radixsort_futures.size(); i++) {
        int ret = radixsort_futures[i].get();
        if (ret == -1) {
            logger_->error("[{}] failed - {}", i, ret);
            FAIL();
        }
    }
    logger_->info("done");

    SUCCEED();
}

}

