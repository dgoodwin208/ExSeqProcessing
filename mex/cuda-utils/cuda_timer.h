#ifndef __CUDA_TIMER_H__
#define __CUDA_TIMER_H__

#include <cuda_runtime.h>

class CudaTimer {
    cudaEvent_t start_;
    cudaEvent_t end_;
    cudaStream_t stream_;

public:
    CudaTimer() : stream_(0) {
        cudaEventCreate(&start_);
        cudaEventCreate(&end_);
        cudaEventRecord(start_, stream_);
    }
    CudaTimer(const cudaStream_t& stream) : stream_(stream) {
        cudaEventCreate(&start_);
        cudaEventCreate(&end_);
        cudaEventRecord(start_, stream_);
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(end_);
    }


    void reset() {
        cudaEventRecord(start_, stream_);
    }
    float get_laptime() {
        cudaEventRecord(end_, stream_);
        cudaEventSynchronize(end_);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start_, end_);
        return milliseconds / 1000.0;
    }
};

#endif // __CUDA_TIMER_H__

