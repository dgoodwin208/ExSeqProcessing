#ifndef __CONVN_IMPL_H__
#define __CONVN_IMPL_H__

#include <string>
#include <tuple>
#include <vector>
#include <future>
#include <thread>

#include "spdlog/spdlog.h"

#define MAX_DATA_SIZE_FOR_GPU 1024*1024*180*sizeof(double)
//#define SEQUENTIAL_RUN

class ConvnImpl {
    public:
        //ConvnImpl();

        void run(float **image, float **kernel, float **h_output);

    protected:
        void waitForTasks(const std::string& task_name, std::vector<std::future<int>>& futures);
};


class ExceptionToMATLAB {
    std::string matlab_id_;
    std::string message_;
public:
    ExceptionToMATLAB(const std::string& matlab_id, const std::string& message)
        : matlab_id_(matlab_id),
          message_(message) {

    }

    const std::string& getMatlabId() const { return matlab_id_; }
    const std::string& getMessage() const { return message_; }
};

#endif // __CONVN_IMPL_H__

