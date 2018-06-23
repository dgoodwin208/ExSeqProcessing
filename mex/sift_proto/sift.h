#ifndef __SIFT_H__
#define __SIFT_H__

#include <sstream>
#include <iomanip>
#include <vector>
#include <memory>
#include <cstdint>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include "cuda_task.h"
#include "sift_types.h"

#include "spdlog/spdlog.h"

#define DEBUG_OUTPUT
//#define DEBUG_OUTPUT_MATRIX
//#define DEBUG_DIST_CHECK
#define DEBUG_NO_THREADING


namespace cudautils {

// pinned memory on host
typedef thrust::host_vector<double, thrust::system::cuda::experimental::pinned_allocator<double>> pinnedDblHostVector;
typedef thrust::host_vector<int8_t, thrust::system::cuda::experimental::pinned_allocator<int8_t>> pinnedInt8HostVector;
typedef thrust::host_vector<unsigned int, thrust::system::cuda::experimental::pinned_allocator<unsigned int>> pinnedUIntHostVector;

inline
unsigned int get_num_blocks(const unsigned int total_size, const unsigned int block_size) {
    return (total_size + block_size - 1) / block_size;
}

inline
unsigned int get_delta(const unsigned int total_size, const unsigned int index, const unsigned int delta) {
    return ((index + 1) * delta < total_size ? delta : total_size - index * delta);
}

inline
__host__ __device__
void ind2sub(const unsigned int x_stride, const unsigned int y_stride, const unsigned int idx, unsigned int& x, unsigned int& y, unsigned int& z) {
    unsigned int i = idx;
    z = i / (x_stride * y_stride);
    i -= z * (x_stride * y_stride);

    y = i / x_stride;
    x = i - y * x_stride;
}

__global__
void create_descriptor(
        unsigned int x_stride,
        unsigned int y_stride,
        unsigned int map_size,
        unsigned int *map_idx,
        int8_t *map,
        double *image,
        cudautils::SiftParams sift_params,
        double* device_centers,
        double *keypoints,
        std::shared_ptr<spdlog::logger> logger_);


class Sub2Ind {

    const unsigned int x_stride_;
    const unsigned int y_stride_;
    const unsigned int z_stride_;

    unsigned int base_x_;
    unsigned int base_y_;
    unsigned int base_z_;

public:
    Sub2Ind(
            const unsigned int x_stride,
            const unsigned int y_stride,
            const unsigned int z_stride,
            const unsigned int base_x = 0,
            const unsigned int base_y = 0,
            const unsigned int base_z = 0)
        : x_stride_(x_stride), y_stride_(y_stride), z_stride_(z_stride),
          base_x_(base_x), base_y_(base_y), base_z_(base_z)
    {
    }

    void setBasePos(const unsigned int base_x, const unsigned int base_y, const unsigned int base_z) {
        assert(base_x < x_stride_);
        assert(base_y < y_stride_);
        assert(base_z < z_stride_);

        base_x_ = base_x;
        base_y_ = base_y;
        base_z_ = base_z;
    }

    size_t operator()(const unsigned int x, const unsigned int y, const unsigned int z) {
        assert(x + base_x_ < x_stride_);
        assert(y + base_y_ < y_stride_);
        assert(z + base_z_ < z_stride_);

        return (size_t(x + base_x_) + size_t(y + base_y_) * size_t(x_stride_) + size_t(z + base_z_) * size_t(x_stride_) * size_t(y_stride_));
    }
};

struct RangeCheck {
    unsigned int x_stride;
    unsigned int y_stride;
    unsigned int x_start;
    unsigned int x_end;
    unsigned int y_start;
    unsigned int y_end;
    unsigned int z_start;
    unsigned int z_end;

    __host__ __device__
    bool operator()(unsigned int idx) {
        unsigned int x;
        unsigned int y;
        unsigned int z;
        ind2sub(x_stride, y_stride, idx, x, y, z);
        return (x >= x_start && x < x_end &&
                y >= y_start && y < y_end &&
                z >= z_start && z < z_end);
    }
};

class Sift : public cudautils::CudaTask {

    //   This class provides a function to interpolate volume image data from the nearest intensities
    //
    //   Input:
    //       volume image data  (x-rows, y-columns, z-slices)
    //       mask map data (1: mask, 0: hole)
    //
    //   Output:
    //       interpolated image data

    const unsigned int x_size_; // rows
    const unsigned int y_size_; // cols
    const unsigned int z_size_;
    const unsigned int x_sub_size_; // rows
    const unsigned int y_sub_size_; // cols
    const unsigned int dx_;
    const unsigned int dy_;
    const unsigned int dw_; // interpolation radius

    const unsigned int num_gpus_;
    const unsigned int num_streams_;

    unsigned int x_sub_stride_;
    unsigned int y_sub_stride_;
    unsigned int dx_stride_;
    unsigned int dy_stride_;
    unsigned int z_stride_; // not divided along the z-axis

    unsigned int num_x_sub_;  // (num_x_sub_-1) * x_sub_size_ < x_ <= num_x_sub_ * x_sub_size_
    unsigned int num_y_sub_;  // (num_y_sub_-1) * y_sub_size_ < y_ <= num_y_sub_ * y_sub_size_


    struct DomainDataOnHost {
        Sub2Ind sub2ind;

        double *h_image;
        int8_t *h_map;
        cudautils::Keypoint_store keystore;
        cudautils::Keypoint *keypoints;

        DomainDataOnHost(
                const unsigned int x_size,
                const unsigned int y_size,
                const unsigned int z_size,
                cudautils::SiftParams sift_params)
            : sub2ind(x_size, y_size, z_size) {
            size_t volume_size = x_size * y_size * z_size;
            cudaHostAlloc(&h_image, volume_size * sizeof(double), cudaHostAllocPortable);
            cudaHostAlloc(&h_map, volume_size * sizeof(int8_t), cudaHostAllocPortable);
            cudaHostAlloc((void**) &keystore, sizeof(cudautils::Keypoint_store), cudaHostAllocPortable);
            cudaHostAlloc(&keypoints, sift_params.keypoint_num * sizeof(cudautils::Keypoint), cudaHostAllocPortable);
            keystore.buf = keypoints;
            keystore.len = sift_params.keypoint_num;
        }
        ~DomainDataOnHost() {
            cudaFreeHost(h_image);
            cudaFreeHost(h_map);
            cudaFreeHost(&(keystore.buf));
            cudaFreeHost(&keystore);
        }
    };

    std::shared_ptr<DomainDataOnHost> dom_data_;


    struct SubDomainDataOnStream {
        cudaStream_t stream;

        Sub2Ind pad_sub2ind;

        std::vector<unsigned int> dx_i_list;
        std::vector<unsigned int> dy_i_list;

        SubDomainDataOnStream(
                const unsigned int dx_stride,
                const unsigned int dy_stride,
                const unsigned int z_stride)
            : pad_sub2ind(dx_stride, dy_stride, z_stride) {
        }
    };


    struct SubDomainDataOnGPU {
        Sub2Ind pad_sub2ind;

        std::vector<unsigned int> x_sub_i_list;
        std::vector<unsigned int> y_sub_i_list;

        double *padded_image;
        int8_t *padded_map;
        unsigned int *padded_map_idx;
        unsigned int padded_map_idx_size;

        std::vector<std::shared_ptr<SubDomainDataOnStream>> stream_data;

        SubDomainDataOnGPU(
                const unsigned int x_sub_stride,
                const unsigned int y_sub_stride,
                const unsigned int z_stride,
                const unsigned int num_streams)
            : pad_sub2ind(x_sub_stride, y_sub_stride, z_stride), stream_data(num_streams) {
            size_t padded_sub_volume_size = x_sub_stride * y_sub_stride * z_stride;
            cudaMalloc(&padded_image,   padded_sub_volume_size * sizeof(double));
            cudaMalloc(&padded_map,     padded_sub_volume_size * sizeof(int8_t));
            cudaMalloc(&padded_map_idx, padded_sub_volume_size * sizeof(unsigned int));
        }

        ~SubDomainDataOnGPU() {
            cudaFree(padded_image);
            cudaFree(padded_map);
            cudaFree(padded_map_idx);
        }
    };

    std::vector<std::shared_ptr<SubDomainDataOnGPU>> subdom_data_;

    std::shared_ptr<spdlog::logger> logger_;

    cudautils::SiftParams sift_params_;

public:
    Sift(
            const unsigned int x_size,
            const unsigned int y_size,
            const unsigned int z_size,
            const unsigned int x_sub_size,
            const unsigned int y_sub_size,
            const unsigned int dx,
            const unsigned int dy,
            const unsigned int dw,
            const unsigned int num_gpus,
            const unsigned int num_streams,
            cudautils::SiftParams sift_params);

    virtual ~Sift();


    void setImage(const double *img);
    void setImage(const std::vector<double>& img);
    void setMapToBeInterpolated(const int8_t *map);
    void setMapToBeInterpolated(const std::vector<int8_t>& map);
    void getKeystore(cudautils::Keypoint_store keystore);
    void getImage(double* img);
    void getImage(std::vector<double>& img);

    virtual int getNumOfGPUTasks(const int gpu_id);
    virtual int getNumOfStreamTasks(const int gpu_id, const int stream_id);

    virtual void prerun() {}
    virtual void postrun() {}

    virtual void runOnGPU(const int gpu_id, const unsigned int gpu_task_id);
    virtual void postrunOnGPU(const int gpu_id, const unsigned int gpu_task_id) {}
    virtual void runOnStream( const int gpu_id, const int stream_id, const unsigned int gpu_task_id);
    cudautils::SiftParams get_sift_params();


}; // class Sift


} // namespace cudautils

#endif // __SIFT_H__

