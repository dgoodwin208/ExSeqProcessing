#include "gtest/gtest.h"

#include <vector>
#include <string>
#include <sstream>
#include <thread>
#include <future>
#include <sys/stat.h>
#include <sys/syscall.h>

#include "gpulock.h"
#include "gpudevice.h"

#include "spdlog/spdlog.h"

namespace {

class CudaUtilsTest : public ::testing::Test {
protected:
    int num_gpus_;
    std::string lock_dir_;
    std::vector<std::string> lock_filenames_;
    std::string all_lock_filename_;

    std::vector<int> count_;
    int num_loops_;
    std::chrono::milliseconds proc_sec_;

    std::shared_ptr<spdlog::logger> logger_;

    CudaUtilsTest() {
        num_gpus_ = cudautils::get_gpu_num();
        logger_ = spdlog::get("console");
        if (logger_ == nullptr) {
            logger_ = spdlog::stdout_logger_mt("console");
        }

        lock_dir_ = "/tmp/.exseqproc";
        for (int i = 0; i < num_gpus_; i++) {
            std::string lock_filename = lock_dir_ + "/gpu" + std::to_string(i) + ".lock";
            lock_filenames_.push_back(lock_filename);

            count_.push_back(0);
        }

        num_loops_ = 0;
    }
    virtual ~CudaUtilsTest() {
    }

    virtual void SetUp() {
        struct stat s;

        if (stat(all_lock_filename_.c_str(), &s) < 0) {
            //logger_->warn("cannot find an all-lock file: {}", all_lock_filename_);
        } else {
            //logger_->info("remove an all-lock file: {}", all_lock_filename_);
            if (remove(all_lock_filename_.c_str()) < 0) {
                logger_->error("cannot remove: {}", all_lock_filename_);
            }
        }

        for (int i = 0; i < num_gpus_; i++) {
            if (stat(lock_filenames_[i].c_str(), &s) < 0) {
                //logger_->warn("cannot find a lock file: {}", lock_filenames_[i]);
                continue;
            }

            //logger_->info("remove a lock file: {}", lock_filenames_[i]);
            if (remove(lock_filenames_[i].c_str()) < 0) {
                logger_->error("cannot remove: {}", lock_filenames_[i]);
            }
        }
    }

    virtual void TearDown() {
        struct stat s;

        if (stat(all_lock_filename_.c_str(), &s) < 0) {
            //logger_->warn("cannot find an all-lock file: {}", all_lock_filename_);
        } else {
            //logger_->info("remove an all-lock file: {}", all_lock_filename_);
            if (remove(all_lock_filename_.c_str()) < 0) {
                logger_->error("cannot remove: {}", all_lock_filename_);
            }
        }

        for (int i = 0; i < num_gpus_; i++) {
            if (stat(lock_filenames_[i].c_str(), &s) < 0) {
                //logger_->warn("cannot find a lock file: {}", lock_filenames_[i]);
                continue;
            }

            //logger_->info("remove a lock file: {}", lock_filenames_[i]);
            if (remove(lock_filenames_[i].c_str()) < 0) {
                logger_->error("cannot remove: {}", lock_filenames_[i]);
            }
        }
    }

public:
    int
    gpulockThread(const std::string& target) {
        logger_->info("[{}] start", target);
        std::ostringstream sout;
        sout << std::this_thread::get_id() << " " << getpid() << " " << syscall(SYS_gettid);
        logger_->info("[{}] tid={}", target, sout.str());

        cudautils::GPULock lock(num_gpus_);
        int gpu_id = -1;

        for (int i = 0; i < num_loops_; i++) {
            gpu_id = lock.trylock(-1);
            if (gpu_id == -1) {
                logger_->info("[{}] error..", target);
                return -1;
            }

            count_[gpu_id]++;
            std::this_thread::sleep_for(proc_sec_);

            if (lock.unlock() < 0) {
                logger_->error("[{}] cannot unlock", target);
                logger_->info("[{}] end", target);
                return -1;
            }
        }

        logger_->info("[{}] end", target);
        return 0;
    }
};

TEST_F(CudaUtilsTest, GPULockAndUnlockTest) {
    int ret;
    cudautils::GPULock lock(num_gpus_);
    ret = lock.trylock();
    ASSERT_GE(ret, 0);

    ret = lock.unlock();
    ASSERT_EQ(0, ret);
}

TEST_F(CudaUtilsTest, GPULockAndBlockTest) {
    int ret;
    std::vector<std::shared_ptr<cudautils::GPULock>> locks;
    for (int i = 0; i < num_gpus_; i++) {
        locks.push_back(std::make_shared<cudautils::GPULock>(num_gpus_));
    }

    for (int i = 0; i < num_gpus_; i++) {
        ret = locks[i]->trylock();
        ASSERT_GE(ret, 0);
    }

    cudautils::GPULock additional_lock(num_gpus_);
    ret = additional_lock.trylock(2);
    ASSERT_EQ(-1, ret);

    for (int i = 0; i < num_gpus_; i++) {
        ret = locks[i]->unlock();
    }
}

TEST_F(CudaUtilsTest, GPULockAllAndUnlockAllTest) {
    int ret;

    cudautils::GPULock lock(num_gpus_);
    ret = lock.trylockall();
    ASSERT_EQ(0, ret);

    ret = lock.unlockall();
    ASSERT_EQ(0, ret);

}

TEST_F(CudaUtilsTest, GPULockAllAndLockOneTest) {
    int ret;

    cudautils::GPULock all_lock(num_gpus_);
    ret = all_lock.trylockall();
    ASSERT_EQ(0, ret);
    // ALL lock: o
    //   lock 0: o
    //   lock 1: o

    cudautils::GPULock one_lock(num_gpus_);
    ret = one_lock.trylock();
    ASSERT_EQ(-1, ret);
    // ALL lock: o      <- trylock one-lock failed
    //   lock 0: o
    //   lock 1: o

    ret = all_lock.unlockall();
    ASSERT_EQ(0, ret);
    // ALL lock: _
    //   lock 0: _
    //   lock 1: _

    ret = one_lock.trylock();
    ASSERT_EQ(0, ret);
    // ALL lock: _
    //   lock 0: o
    //   lock 1: _

    ret = all_lock.trylockall();
    ASSERT_EQ(-1, ret);
    // ALL lock: _      <- trylock all-lock failed
    //   lock 0: o
    //   lock 1: _

    ret = one_lock.unlock();
    ASSERT_EQ(0, ret);
    // ALL lock: _
    //   lock 0: _
    //   lock 1: _
}

TEST_F(CudaUtilsTest, GPULockMultiThread1Test) {
    const size_t num_parallel = 200;
    num_loops_ = 5;
    proc_sec_ = std::chrono::milliseconds(0);
    std::vector<std::future<int>> futures;

    for (int i = 0; i < num_gpus_; i++) {
        count_[i] = 0;
    }

    for (size_t i = 0; i < num_parallel; i++) {
        std::string target = "t" + std::to_string(i);
        futures.push_back(std::async(std::launch::async,
                    &CudaUtilsTest::gpulockThread, this, target));
    }

    for (size_t i = 0; i < futures.size(); i++) {
        int ret = futures[i].get();
        if (ret == -1) {
            logger_->error("[{}] failed - {}", i, ret);
            FAIL();
        }
    }

    int sum = 0;
    for (int i = 0; i < num_gpus_; i++) {
        logger_->info("count[{}] = {}", i, count_[i]);
        sum += count_[i];
    }
    logger_->info("done");

    ASSERT_EQ(num_parallel * num_loops_, sum);
}

TEST_F(CudaUtilsTest, GPULockMultiThread2Test) {
    const size_t num_parallel = 20;
    num_loops_ = 5;
    proc_sec_ = std::chrono::milliseconds(10);
    std::vector<std::future<int>> futures;

    for (int i = 0; i < num_gpus_; i++) {
        count_[i] = 0;
    }

    for (size_t i = 0; i < num_parallel; i++) {
        std::string target = "t" + std::to_string(i);
        futures.push_back(std::async(std::launch::async,
                    &CudaUtilsTest::gpulockThread, this, target));
    }

    for (size_t i = 0; i < futures.size(); i++) {
        int ret = futures[i].get();
        if (ret == -1) {
            logger_->error("[{}] failed - {}", i, ret);
            FAIL();
        }
    }

    int sum = 0;
    for (int i = 0; i < num_gpus_; i++) {
        logger_->info("count[{}] = {}", i, count_[i]);
        sum += count_[i];
    }
    logger_->info("done");

    ASSERT_EQ(num_parallel * num_loops_, sum);
}

//TODO MultiProcessesTest

}

