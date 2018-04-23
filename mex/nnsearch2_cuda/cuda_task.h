#ifndef __CUDA_TASK_H__
#define __CUDA_TASK_H__

#include <vector>
#include <memory>

namespace parallelutils {

class CudaTask {

public:
    CudaTask() {}
    virtual ~CudaTask() {}

    virtual int getNumOfGPUTasks(const int gpu_id) = 0;
    virtual int getNumOfStreamTasks(const int gpu_id, const int stream_id) = 0;

    virtual void prerun() = 0;
    virtual void postrun() = 0;

    virtual void runOnGPU(const int gpu_id, const unsigned int gpu_task_id) = 0;
    virtual void postrunOnGPU(const int gpu_id, const unsigned int gpu_task_id) = 0;
    virtual void runOnStream( const int gpu_id, const int stream_id, const unsigned int gpu_task_id) = 0;

};

}

#endif

