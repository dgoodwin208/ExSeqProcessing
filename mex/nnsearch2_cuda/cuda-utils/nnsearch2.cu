#include <iostream>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>


#include <cuda_runtime.h>

#include "nnsearch2.h"
#include "defaults.h"
#include "cuda_timer.h"

#include "spdlog/spdlog.h"


namespace cudautils {

// summation of squared values of each vector in matrix
//
//     |<------  k  ----->|
//     [x1,1 x1,2 ... x1,k]        [x1,1^2 + x1,1^2 + ... + x1,k^2]
// x = [x2,1 x2,2 ... x2,k]   -->  [x2,1^2 + x2,2^2 + ... + x2,k^2]
//                ...                                 ...
//     [xw,1 xw,2 ... xw,k]        [xw,1^2 + xw,2^2 + ... + xw,k^2]
//
__global__
void sum_squared(
        unsigned int w,
        unsigned int k,
        double *x,
        double *x2) {

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= w) return;

    double sum = 0.0;
    for (unsigned int v = 0; v < k; v++) {
        sum += x[i + v * w] * x[i + v * w];
    }
    x2[i] = sum;
}

// calculation of squared norms
//
// ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x <x, y>
//
__global__
void calc_squared_norm(
        unsigned int m,
        unsigned int n,
        unsigned int k,
        double *x,
        double *y,
        double *x2,
        double *y2,
        double *r) {

    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;

    double norm = x2[i] + y2[j];
    for (unsigned int w = 0; w < k; w++) {
        norm += -2.0 * x[i + w * m] * y[j + w * n];
    }
    r[j + n * i] = norm;
    //TODO use shared memory and memory coalescing
    // this code uses L2 cache that is beating shared memory...
}

__global__
void calc_squared_norm2(
        unsigned int m,
        unsigned int n,
        unsigned int k,
        double *x,
        double *y,
        double *x2,
        double *y2,
        double *r) {

    const unsigned int block_size = defaults::num_threads_in_calc_sqnorm_func;
    const unsigned int num_sx_blocks = 3;
    __shared__ double sx[block_size * num_sx_blocks];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x + blockIdx.y * gridDim.x;

    if (i >= m) return;

    for (unsigned int u = 0; u < num_sx_blocks; u++) {
        unsigned int sx_w = tid + u * block_size;
        if (sx_w < k) {
            sx[sx_w] = x[i + sx_w * m];
        } else {
            break;
        }
    }
    __syncthreads();

    unsigned int x2_val = x2[i];
    for (unsigned int j = 0; j < n; j += block_size) {
        if (tid + j < n) {
            double norm = x2_val + y2[tid + j];
            for (unsigned int w = 0; w < k; w++) {
                norm += -2.0 * sx[w] * y[tid + j + w * n];
            }

            r[tid + j + i * n] = norm;
        }
    }
}

// get two minimums from each block of vectors
//
//      |<------  m  ----->|
//      [r1,1 r1,2 ... r1,m]
// r0 = [r2,1 r2,2 ... r2,m]   -->  r = [ra,1 rc,2 + ... + ry,m]  idx = [a c ... y]
//                ...                   [rb,1 rd,2 + ... + rz,m]        [b d ... z]
//      [rw,1 rw,2 ... rn,m]
//
__global__
void get_two_mins(
        unsigned int n,
        unsigned int idx_start,
        double *r0,
        double *r,
        unsigned int *idx) {

    __shared__ double sdist2[2 * defaults::num_threads_in_twotops_func];
    __shared__ unsigned int sindex[2 * defaults::num_threads_in_twotops_func];
    // num_threads must be a power of 2.

    unsigned int tid  = threadIdx.x;
    unsigned int tid2 = threadIdx.x + defaults::num_threads_in_twotops_func;
    unsigned int i1 = tid  + blockIdx.x * 2 * defaults::num_threads_in_twotops_func;
    unsigned int i2 = tid2 + blockIdx.x * 2 * defaults::num_threads_in_twotops_func;
    unsigned int block_id = blockIdx.y + blockIdx.z * gridDim.y;

    sdist2[tid ] = (i1 < n) ? r0[i1 + block_id * n] : UINT_MAX;
    sdist2[tid2] = (i2 < n) ? r0[i2 + block_id * n] : UINT_MAX;
    sindex[tid ] = (i1 < n) ? i1 + idx_start : UINT_MAX;
    sindex[tid2] = (i2 < n) ? i2 + idx_start : UINT_MAX;
    __syncthreads();

    if (i1 >= n) return;

    // basically, two minimum values are selected from four values using 4 steps below
    // [a, b, c, d] --> [a, b] < [c, d]
    //
    // 1) if a > c, then swap a and c   [a, b, c, d] --> [a] < [c]; [b, d]
    // 2) if b > d, then swap b and d                --> [a] < [c]; [b] < [d]
    // 3) if b > c, then swap b and c                --> [a, b] < [c, d]
    // 4) if b <= c                                  --> [a, b] < [c]; [b] < [d]   (not sure if a < d or d < a)
    //       && a > d, then swap a and d             --> [a, b] < [c, d]
    //
    // simple case: v=1, w=2
    // addr.0  =0 | a
    // addr.v  =1 | b
    // addr.0+w=2 | c
    // addr.v+w=3 | d

    for (unsigned int w = defaults::num_threads_in_twotops_func; w >= 2; w >>= 1) {
        if (tid < w) {
            // a = sdist2[tid], c = sdist2[tid + w]
            if (sdist2[tid] > sdist2[tid + w]) {
                thrust::swap(sdist2[tid], sdist2[tid + w]);
                thrust::swap(sindex[tid], sindex[tid + w]);
            }
        }
        __syncthreads();

        unsigned int v = w/2;
        if (tid < v) {
            // a = sdist2[tid], c = sdist2[tid + w]
            // b = sdist2[tid + v], d = sdist2[tid + w + v]
            if (sdist2[tid + w] < sdist2[tid + v]) {
                thrust::swap(sdist2[tid + w], sdist2[tid + v]);
                thrust::swap(sindex[tid + w], sindex[tid + v]);
            } else if (sdist2[tid + w + v] < sdist2[tid]) {
                thrust::swap(sdist2[tid + w + v], sdist2[tid]);
                thrust::swap(sindex[tid + w + v], sindex[tid]);
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        r[2 * blockIdx.x     + block_id * 2 * gridDim.x] = sdist2[0];
        r[2 * blockIdx.x + 1 + block_id * 2 * gridDim.x] = sdist2[1];

        idx[2 * blockIdx.x     + block_id * 2 * gridDim.x] = sindex[0];
        idx[2 * blockIdx.x + 1 + block_id * 2 * gridDim.x] = sindex[1];
    }
}

__global__
void get_two_mins_with_index(
        const unsigned int stride,
        const unsigned int n,
        const unsigned int m,
        double *x,
        unsigned int *idx) {

    __shared__ double sdist2[2 * defaults::num_threads_in_twotops_func];
    __shared__ unsigned int sindex[2 * defaults::num_threads_in_twotops_func];
    // num_threads must be a power of 2.

    unsigned int tid  = threadIdx.x;
    unsigned int tid2 = threadIdx.x + defaults::num_threads_in_twotops_func;
    unsigned int i1 = tid  + blockIdx.x * 2 * defaults::num_threads_in_twotops_func;
    unsigned int i2 = tid2 + blockIdx.x * 2 * defaults::num_threads_in_twotops_func;
    unsigned int block_id = blockIdx.y + blockIdx.z * gridDim.y;

    if (block_id >= m) return;

    sdist2[tid ] = (i1 < n) ? x  [i1 + block_id * stride] : UINT_MAX;
    sdist2[tid2] = (i2 < n) ? x  [i2 + block_id * stride] : UINT_MAX;
    sindex[tid ] = (i1 < n) ? idx[i1 + block_id * stride] : UINT_MAX;
    sindex[tid2] = (i2 < n) ? idx[i2 + block_id * stride] : UINT_MAX;
    __syncthreads();

    if (i1 >= n) return;

    for (unsigned int w = defaults::num_threads_in_twotops_func; w >= 2; w >>= 1) {
        if (tid < w) {
            // a = sdist2[tid], b = sdist2[tid + w]
            if (sdist2[tid] > sdist2[tid + w]) {
                thrust::swap(sdist2[tid], sdist2[tid + w]);
                thrust::swap(sindex[tid], sindex[tid + w]);
            }
        }
        __syncthreads();

        unsigned int v = w/2;
        if (tid < v) {
            // a = sdist2[tid], b = sdist2[tid + w]
            // c = sdist2[tid + v], d = sdist2[tid + w + v]
            if (sdist2[tid + w] < sdist2[tid + v]) {
                thrust::swap(sdist2[tid + w], sdist2[tid + v]);
                thrust::swap(sindex[tid + w], sindex[tid + v]);
            } else if (sdist2[tid + w + v] < sdist2[tid]) {
                thrust::swap(sdist2[tid + w + v], sdist2[tid]);
                thrust::swap(sindex[tid + w + v], sindex[tid]);
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        x[i1     + block_id * stride] = sdist2[0];
        x[i1 + 1 + block_id * stride] = sdist2[1];

        idx[i1     + block_id * stride] = sindex[0];
        idx[i1 + 1 + block_id * stride] = sindex[1];
    }
}

// gather values on blocks
//       summary of size               val, idx                          val, idx
//      _____________         ____________________________      ____________________________
//       |   |    |         b1| b1,1.top1 | b1,2.top1 |         | b1,1.top1 | b1,2.top1 |
//       |   | block_size     | b1,1.top2 | b1,2.top2 |         | b1,1.top2 | b1,2.top2 |
//       |   |____|__         |___________|___________|___      | b2,1.top1 | b2,2.top1 |
//       |   |              b2| b2,1.top1 | b2,2.top1 |         | b2,1.top2 | b2,2.top2 |
//       |   |                | b2,1.top2 | b2,2.top2 |         | b3,1.top1 | b3,2.top1 |
//       |   |_______         |___________|___________|___      | b3,1.top2 | b3,2.top2 |
//       |   |              b3| b3,1.top1 | b3,2.top1 |         |     ...   |    ...    |
//       |   |                |           |           |    -->  |           |           |
//       |   |   ...          |   ...     |   ...     |         | bn,1.top1 | bn,2.top1 |
//       |   |_______         |___________|___________|___      | bn,1.top2 | bn,2.top2 |
//       |   |              bn| bn,1.top1 | bn,2.top1 |         |-----------|-----------|---
//       | n_size             | bn,1.top2 | bn,2.top2 |         |           |           |
//       |___|_______         |___________|___________|___      |           |           |
//       |                    |           |           |         |    N/A    |    N/A    |
//      stride                |    N/A    |    N/A    |         |           |           |
//      _|___________         |___________|___________|___      |___________|___________|___
//
__global__
void gather_values_on_blocks(
        const unsigned int stride,
        const unsigned int n_size,
        const unsigned int block_size,
        const unsigned int m,
        double *val,
        unsigned int* idx) {

    __shared__ double sval[2 * defaults::num_threads_in_twotops_func];
    __shared__ unsigned int sindex[2 * defaults::num_threads_in_twotops_func];

    unsigned int tid = threadIdx.x;
    unsigned int block_id = blockIdx.x + blockIdx.y * gridDim.x;

    if (block_id >= m) return;

    for (unsigned int base_idx = 0; base_idx < n_size; base_idx += block_size) {
        unsigned int i = tid + base_idx;
        unsigned int j = (i / 2) * block_size + (i % 2);
        if (i >= n_size || j >= n_size) {
            continue;
        }

        sval[tid]   = val[j + block_id * stride];
        sindex[tid] = idx[j + block_id * stride];
        __syncthreads();

        val[i + block_id * stride] = sval[tid];
        idx[i + block_id * stride] = sindex[tid];
    }
}

// swap sort
//       size             val, idx                      val, idx
//      ______     ________________________      ________________________
//       |         | b1.val2 | b2.val4 |         | b1.val1 | b2.val3 |
//       |         | b1.val1 | b2.val3 |         | b1.val2 | b2.val4 |
//       |         |---------|---------|---      |---------|---------|---
//       |         |         |         |    -->  |         |         |
//      stride     |   N/A   |   N/A   |         |   N/A   |   N/A   |
//      _|____     |_________|_________|___      |_________|_________|___
//                   b1.val2 > b1.val1, b2.val4 > b2.val3
//
__global__
void swap_sort(
        const unsigned int stride,
        const unsigned int total_size,
        double *val,
        unsigned int *idx) {

    unsigned int i = (threadIdx.x + blockIdx.x * blockDim.x) * stride;

    if (i >= total_size) return;

    if (val[i] > val[i + 1]) {
        thrust::swap(val[i], val[i + 1]);
        thrust::swap(idx[i], idx[i + 1]);
    }
}



NearestNeighborSearch::NearestNeighborSearch(
        const unsigned int m,
        const unsigned int n,
        const unsigned int k,
        const unsigned int dm,
        const unsigned int dn,
        const unsigned int num_gpus,
        const unsigned int num_streams)
: m_(m), n_(n), k_(k), dm_(dm), dn_(dn),
    num_gpus_(num_gpus), num_streams_(num_streams)
{
    logger_ = spdlog::get("console");
    if (! logger_) {
        logger_ = spdlog::stdout_logger_mt("console");
    }
#ifdef DEBUG_OUTPUT
    spdlog::set_level(spdlog::level::debug);
#else
    logger_->set_level(spdlog::level::off);
#endif

    size_t log_q_size = 4096;
    spdlog::set_async_mode(log_q_size);

    num_dm_ = get_num_blocks(m_, dm_);
    num_dn_ = get_num_blocks(n_, dn_);
#ifdef DEBUG_OUTPUT
    logger_->info("m={}, dm={}, num_dm={}", m_, dm_, num_dm_);
    logger_->info("n={}, dn={}, num_dn={}", n_, dn_, num_dn_);
#endif

    n_blocks_in_two_mins_ = get_num_blocks(dn_, defaults::num_threads_in_twotops_func * 2);

    dom_data_ = std::make_shared<DomainDataOnHost>(m_, n_, k_, num_dn_);
    subdom_data_.resize(num_gpus_);
    for (unsigned int i = 0; i < num_gpus_; i++) {
        cudaSetDevice(i);

        subdom_data_[i] = std::make_shared<SubDomainDataOnGPU>();
        subdom_data_[i]->y.resize(dn_ * k_);
        subdom_data_[i]->y2.resize(dn_);

        subdom_data_[i]->stream_data.resize(num_streams_);

        for (unsigned int j = 0; j < num_streams_; j++) {
            subdom_data_[i]->stream_data[j] = std::make_shared<SubDomainDataOnStream>();
            subdom_data_[i]->stream_data[j]->x.resize(dm_ * k_);
            subdom_data_[i]->stream_data[j]->x2.resize(dm_);
            subdom_data_[i]->stream_data[j]->r.resize(dm_ * dn_);

            subdom_data_[i]->stream_data[j]->val.resize(2 * n_blocks_in_two_mins_ * dm_);
            subdom_data_[i]->stream_data[j]->idx.resize(2 * n_blocks_in_two_mins_ * dm_);

            cudaStreamCreate(&subdom_data_[i]->stream_data[j]->stream);
        }
    }
    cudaSetDevice(0);

    for (unsigned int y_i = 0; y_i < num_dn_; y_i++) {
        unsigned int gpu_id = y_i % num_gpus_;
        subdom_data_[gpu_id]->y_i_list.push_back(y_i);
    }
}

NearestNeighborSearch::~NearestNeighborSearch() {
    for (unsigned int i = 0; i < num_gpus_; i++) {
        for (unsigned int j = 0; j < num_streams_; j++) {
            cudaStreamDestroy(subdom_data_[i]->stream_data[j]->stream);
        }
    }

    //logger_->flush();
}

void NearestNeighborSearch::generateSequences() {
#ifdef DEBUG_OUTPUT
    CudaTimer timer;
#endif
    thrust::sequence(dom_data_->h_x.begin(), dom_data_->h_x.end());
    thrust::sequence(dom_data_->h_y.begin(), dom_data_->h_y.end(), 1);
#ifdef DEBUG_OUTPUT
    logger_->info("generate_sequences (host) {}", timer.get_laptime());
#endif
}

void NearestNeighborSearch::setInput(double* in_x, double* in_y) {
    thrust::copy(in_x, in_x + m_ * k_, dom_data_->h_x.begin());
    thrust::copy(in_y, in_y + n_ * k_, dom_data_->h_y.begin());
}

void NearestNeighborSearch::setInput(std::vector<double>& in_x, std::vector<double>& in_y) {
    thrust::copy(in_x.begin(), in_x.end(), dom_data_->h_x.begin());
    thrust::copy(in_y.begin(), in_y.end(), dom_data_->h_y.begin());
}

void NearestNeighborSearch::getResult(double* out_mins_val, unsigned int* out_mins_idx) {
    for (unsigned int j = 0; j < 2; j++) {
        for (unsigned int i = 0; i < m_; i++) {
            out_mins_val[i + j * m_] = dom_data_->h_mins_val[j + i * 2];
            out_mins_idx[i + j * m_] = dom_data_->h_mins_idx[j + i * 2] + 1;
        }
    }
}

void NearestNeighborSearch::getResult(std::vector<double>& out_mins_val, std::vector<unsigned int>& out_mins_idx) {
    for (unsigned int j = 0; j < 2; j++) {
        for (unsigned int i = 0; i < m_; i++) {
            out_mins_val[i + j * m_] = dom_data_->h_mins_val[j + i * 2];
            out_mins_idx[i + j * m_] = dom_data_->h_mins_idx[j + i * 2] + 1;
        }
    }
}

void NearestNeighborSearch::getDist2(double** out_dist2) {
#ifdef DEBUG_DIST_CHECK
    for (size_t i = 0; i < m_; i++) {
        for (size_t j = 0; j < n_; j++) {
            (*out_dist2)[j + i * n_] = dom_data_->h_r[j + i * n_];
        }
    }
#endif
}

double NearestNeighborSearch::getDist2(size_t i, size_t j) {
#ifdef DEBUG_DIST_CHECK
    return dom_data_->h_r[j + i * n_];
#else
    return 0.0;
#endif
}

int NearestNeighborSearch::getNumOfGPUTasks(const int gpu_id) {
    return subdom_data_[gpu_id]->y_i_list.size();
}

int NearestNeighborSearch::getNumOfStreamTasks(
        const int gpu_id,
        const int stream_id) {
    return 1;
}

void NearestNeighborSearch::postrun() {

    try {

        if (num_dn_ > 1) {
#ifdef DEBUG_OUTPUT
            CudaTimer timer;
            logger_->info("===== get total two tops of mins");
#endif
            getTotalTwoTopsOfMins();
#ifdef DEBUG_OUTPUT
            logger_->info("get total two tops of mins ", timer.get_laptime());
#endif
        }
    } catch (...) {
        logger_->error("postrun() has exception.");
    }
}

void NearestNeighborSearch::runOnGPU(
        const int gpu_id,
        const unsigned int gpu_task_id) {

    cudaSetDevice(gpu_id);

    std::shared_ptr<SubDomainDataOnGPU> subdom_data = subdom_data_[gpu_id];

    unsigned int y_i = subdom_data->y_i_list[gpu_task_id];

    unsigned int y_start = y_i * dn_;
    unsigned int n_steps = get_delta(n_, y_i, dn_);
#ifdef DEBUG_OUTPUT
    logger_->info("y_i={},n_steps={}", y_i, n_steps);
#endif

    precacheSquaredDistance(n_, k_, n_steps, y_start,
            dom_data_->h_y, subdom_data->y, subdom_data->y2, subdom_data->stream_data[0]->stream);

    cudaStreamSynchronize(subdom_data->stream_data[0]->stream);

    for (unsigned int x_i = 0; x_i < num_dm_; x_i++) {
        unsigned int s_i = x_i % num_streams_;
#ifdef DEBUG_OUTPUT
        logger_->info("x_i={},y_i={}", x_i, y_i);
#endif

        subdom_data->stream_data[s_i]->x_i_list.push_back(x_i);
    }
}

void NearestNeighborSearch::runOnStream(
        const int gpu_id,
        const int stream_id,
        const unsigned int gpu_task_id) {

    cudaSetDevice(gpu_id);

    std::shared_ptr<SubDomainDataOnGPU> subdom_data = subdom_data_[gpu_id];
    std::shared_ptr<SubDomainDataOnStream> stream_data = subdom_data->stream_data[stream_id];

    unsigned int y_i = subdom_data->y_i_list[gpu_task_id];

#ifdef DEBUG_OUTPUT
    CudaTimer timer(stream_data->stream);
#endif

    unsigned int y_start = y_i * dn_;
    unsigned int n_steps = get_delta(n_, y_i, dn_);

    for (auto it = stream_data->x_i_list.begin(); it != stream_data->x_i_list.end(); ++it) {
        unsigned int x_i = *it;

        unsigned int x_start = x_i * dm_;
        unsigned int m_steps = get_delta(m_, x_i, dm_);
#ifdef DEBUG_OUTPUT
        logger_->info("gpu_id={},stream_id={},y_i={},x_i={},m_steps={},n_steps={}", gpu_id, stream_id, y_i, x_i, m_steps, n_steps);
        logger_->info("setup dist2 {}", timer.get_laptime());
        timer.reset();
        logger_->info("===== calc dist2");
#endif
        calcSquaredDistanceWithCachedY(m_, k_, m_steps, n_steps, x_start, y_start,
                dom_data_->h_x,
                stream_data->x, stream_data->x2,
                subdom_data->y, subdom_data->y2, stream_data->r,
                stream_data->stream);
#ifdef DEBUG_OUTPUT
        logger_->info("calc dist2 {}", timer.get_laptime());
        timer.reset();
        logger_->info("===== get two tops of mins");
#endif
        getTwoTopsOfMinsInBlock(gpu_id, stream_id, x_i, y_i, m_steps, n_steps, y_start);
#ifdef DEBUG_OUTPUT
        logger_->info("get two tops of mins {}", timer.get_laptime());
#endif

        cudaStreamSynchronize(stream_data->stream);

#ifdef DEBUG_DIST_CHECK
#ifdef DEBUG_OUTPUT
        timer.reset();
#endif
        size_t r_i = size_t(y_start) + size_t(x_start) * size_t(n_);
        cudaMemcpy2DAsync(thrust::raw_pointer_cast(&dom_data_->h_r[r_i]), n_ * sizeof(double),
                thrust::raw_pointer_cast(stream_data->r.data()), n_steps * sizeof(double), n_steps * sizeof(double), m_steps,
                cudaMemcpyDeviceToHost, stream_data->stream);
#ifdef DEBUG_OUTPUT
        logger_->info("transfer d2h {}", timer.get_laptime());
#endif
#endif
    }
}

void NearestNeighborSearch::precacheSquaredDistance(
        const unsigned int n,
        const unsigned int k,
        const unsigned int dn,
        const unsigned int y_start,
        pinnedDblHostVector& h_y,
        thrust::device_vector<double>& y,
        thrust::device_vector<double>& y2,
        cudaStream_t& stream) {

#ifdef DEBUG_OUTPUT
    CudaTimer timer(stream);
#endif

    cudaMemcpy2DAsync(thrust::raw_pointer_cast(y.data()), dn * sizeof(double),
            thrust::raw_pointer_cast(&h_y[y_start]), n * sizeof(double), dn * sizeof(double), k,
            cudaMemcpyHostToDevice, stream);

    unsigned int n_blocks = get_num_blocks(dn, 1024);

    sum_squared<<<n_blocks, 1024, 0, stream>>>(dn, k,
            thrust::raw_pointer_cast(y.data()), thrust::raw_pointer_cast(y2.data()));
    cudaStreamSynchronize(stream);
#ifdef DEBUG_OUTPUT
    logger_->info("sum_sqrd y {}", timer.get_laptime());
#endif
}

void NearestNeighborSearch::calcSquaredDistanceWithCachedY(
        const unsigned int m, const unsigned int k,
        const unsigned int dm, const unsigned int dn,
        const unsigned int x_start, const unsigned int y_start,
        pinnedDblHostVector& h_x,
        thrust::device_vector<double>& x,
        thrust::device_vector<double>& x2,
        thrust::device_vector<double>& y,
        thrust::device_vector<double>& y2,
        thrust::device_vector<double>& r,
        cudaStream_t& stream) {

#ifdef DEBUG_OUTPUT
    CudaTimer sub_total_timer(stream);
    CudaTimer timer(stream);
#endif

    cudaMemcpy2DAsync(thrust::raw_pointer_cast(x.data()), dm * sizeof(double),
            thrust::raw_pointer_cast(&h_x[x_start]), m * sizeof(double), dm * sizeof(double), k,
            cudaMemcpyHostToDevice, stream);

    unsigned int m_blocks = get_num_blocks(dm, 1024);
    unsigned int n_blocks = get_num_blocks(dn, 1024);
#ifdef DEBUG_OUTPUT
    logger_->info("setup (device) {}", timer.get_laptime());
    timer.reset();
#endif

    sum_squared<<<m_blocks, 1024, 0, stream>>>(dm, k,
            thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(x2.data()));
#ifdef DEBUG_OUTPUT
    logger_->info("sum_sqrd x {}", timer.get_laptime());
    timer.reset();
#endif

    m_blocks = get_num_blocks(dm, 32);
    n_blocks = get_num_blocks(dn, 32);
    dim3 blocks_norm = dim3(n_blocks, m_blocks, 1);
    dim3 threads_norm = dim3(32, 32, 1);
#ifdef DEBUG_OUTPUT
    logger_->info("m_blocks={},n_blocks={}", m_blocks, n_blocks);
#endif

    calc_squared_norm<<<blocks_norm, threads_norm, 0, stream>>>(dm, dn, k,
        thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(y.data()),
        thrust::raw_pointer_cast(x2.data()), thrust::raw_pointer_cast(y2.data()),
        thrust::raw_pointer_cast(r.data()));
#ifdef DEBUG_OUTPUT
    logger_->info("calc sqrd norm {}", timer.get_laptime());
    logger_->info("sub total {}", sub_total_timer.get_laptime());
    logger_->info("=====");
#endif
}

void NearestNeighborSearch::getTwoTopsOfMinsInBlock(
        const unsigned int gpu_id,
        const unsigned int s_i,
        const unsigned int x_i,
        const unsigned int y_i,
        const unsigned int m_steps,
        const unsigned int n_steps,
        const unsigned int y_start) {

    std::shared_ptr<SubDomainDataOnGPU> subdom_data = subdom_data_[gpu_id];
    std::shared_ptr<SubDomainDataOnStream> stream_data = subdom_data->stream_data[s_i];

#ifdef DEBUG_OUTPUT
    CudaTimer timer;
#endif
    unsigned int n_block_size = 2 * defaults::num_threads_in_twotops_func;
    unsigned int n_blocks = get_num_blocks(n_steps, n_block_size);
    dim3 blocks_two_mins(n_blocks, m_steps, 1);

#ifdef DEBUG_OUTPUT
    logger_->info("== get two tops of mins (1-1)");
    logger_->info("n_blocks={},m_steps={},n_steps={},y_start={}", n_blocks, m_steps, n_steps, y_start);
#endif

    get_two_mins<<<blocks_two_mins, defaults::num_threads_in_twotops_func, 0, stream_data->stream>>>(
            n_steps,
            y_start,
            thrust::raw_pointer_cast(stream_data->r.data()),
            thrust::raw_pointer_cast(stream_data->val.data()),
            thrust::raw_pointer_cast(stream_data->idx.data()));
#ifdef DEBUG_OUTPUT
    logger_->info("get two tops of mins (1-1) {}", timer.get_laptime());
    timer.reset();
    logger_->info("== get two tops of mins (1-2)");
#endif

    unsigned int n_stride = 2 * n_blocks;
    unsigned int cur_n_size = 2 * n_blocks;
    unsigned int count = 1;
    while (n_blocks > 1) {
        n_blocks = get_num_blocks(cur_n_size, n_block_size);
#ifdef DEBUG_OUTPUT
        logger_->info("[{}] n_stride={},n_blocks={},cur_n_size={}", count++, n_stride, n_blocks, cur_n_size);
#endif

        //CAUTION: n_blocks and m_steps should be under 65535
        blocks_two_mins = dim3(n_blocks, m_steps, 1);

        get_two_mins_with_index<<<blocks_two_mins, defaults::num_threads_in_twotops_func, 0, stream_data->stream>>>(
                n_stride,
                cur_n_size,
                m_steps,
                thrust::raw_pointer_cast(stream_data->val.data()),
                thrust::raw_pointer_cast(stream_data->idx.data()));

        gather_values_on_blocks<<<m_steps, n_block_size, 0, stream_data->stream>>>(
                n_stride,
                cur_n_size,
                n_block_size,
                m_steps,
                thrust::raw_pointer_cast(stream_data->val.data()),
                thrust::raw_pointer_cast(stream_data->idx.data()));

        cur_n_size = 2 * n_blocks;

        if (count > 100) {
            std::cout << "unexpected many loops.." << std::endl;
            break;
        }
    }
#ifdef DEBUG_OUTPUT
    logger_->info("get two tops of mins (1-2) {}", timer.get_laptime());
#endif

    if (num_dn_ == 1) {
        unsigned int num_dm_blocks = get_num_blocks(m_steps, defaults::num_threads_in_swap_sort_func);
#ifdef DEBUG_OUTPUT
        logger_->info("===== swap sort (1)");
        logger_->info("n_stride={},total={},num_dm_blocks={}", n_stride, n_stride * m_steps, num_dm_blocks);
        timer.reset();
#endif

        swap_sort<<<num_dm_blocks, defaults::num_threads_in_swap_sort_func, 0, stream_data->stream>>>(
                n_stride,
                n_stride * m_steps,
                thrust::raw_pointer_cast(stream_data->val.data()),
                thrust::raw_pointer_cast(stream_data->idx.data()));

#ifdef DEBUG_OUTPUT
        logger_->info("swap sort (1) {}", timer.get_laptime());
#endif
    }

#ifdef DEBUG_OUTPUT
    timer.reset();
#endif
    cudaMemcpy2DAsync(thrust::raw_pointer_cast(&dom_data_->h_mins_val[2 * y_i + x_i * 2 * num_dn_ * dm_]), 2 * num_dn_ * sizeof(double),
            thrust::raw_pointer_cast(stream_data->val.data()), n_stride * sizeof(double), 2 * sizeof(double), m_steps,
            cudaMemcpyDeviceToHost, stream_data->stream);
#ifdef DEBUG_OUTPUT
    logger_->info("copy dev to host (val) {}", timer.get_laptime());
    timer.reset();
#endif

    cudaMemcpy2DAsync(thrust::raw_pointer_cast(&dom_data_->h_mins_idx[2 * y_i + x_i * 2 * num_dn_ * dm_]), 2 * num_dn_ * sizeof(unsigned int),
            thrust::raw_pointer_cast(stream_data->idx.data()), n_stride * sizeof(unsigned int), 2 * sizeof(unsigned int), m_steps,
            cudaMemcpyDeviceToHost, stream_data->stream);
#ifdef DEBUG_OUTPUT
    logger_->info("copy dev to host (idx) {}", timer.get_laptime());
#endif

    cudaStreamSynchronize(stream_data->stream);
}

void NearestNeighborSearch::getTotalTwoTopsOfMins()
{
#ifdef DEBUG_OUTPUT
    CudaTimer timer;
    logger_->info("===== gather results in sub domains");
#endif

    thrust::device_vector<double> val(dom_data_->h_mins_val);
    thrust::device_vector<unsigned int> idx(dom_data_->h_mins_idx);

    unsigned int n_block_size = 2 * defaults::num_threads_in_twotops_func;
    unsigned int n_stride = 2 * num_dn_;
    unsigned int cur_n_size = 2 * num_dn_;
    unsigned int next_n_blocks;
    unsigned int count = 1;
    do {
#ifdef DEBUG_OUTPUT
        logger_->info("== get two tops of min (2)");
#endif

        next_n_blocks = get_num_blocks(cur_n_size, n_block_size);
#ifdef DEBUG_OUTPUT
        logger_->info("[{}] n_stride={},next_n_blocks={},cur_n_size={}", count++, n_stride, next_n_blocks, cur_n_size);
#endif

        unsigned int num_m_blocks_z = get_num_blocks(m_, defaults::num_blocks_y_in_twotops_func);
        unsigned int num_m_blocks_y = (num_m_blocks_z == 1) ? m_ : defaults::num_blocks_y_in_twotops_func;
#ifdef DEBUG_OUTPUT
        logger_->info("[{}] num_m_blocks_y={},num_m_blocks_z={}", count, num_m_blocks_y, num_m_blocks_z);
        logger_->info("[{}] val_size={}", count, val.size());
#endif

        dim3 blocks_two_mins(next_n_blocks, num_m_blocks_y, num_m_blocks_z);

        get_two_mins_with_index<<<blocks_two_mins, defaults::num_threads_in_twotops_func, 0, 0>>>(
                n_stride,
                cur_n_size,
                m_,
                thrust::raw_pointer_cast(val.data()),
                thrust::raw_pointer_cast(idx.data()));

        dim3 blocks_gather(num_m_blocks_y, num_m_blocks_z, 1);

        gather_values_on_blocks<<<blocks_gather, n_block_size>>>(
                n_stride,
                cur_n_size,
                n_block_size,
                m_,
                thrust::raw_pointer_cast(val.data()),
                thrust::raw_pointer_cast(idx.data()));

        cur_n_size = 2 * next_n_blocks;

        if (count > 100) {
            logger_->error("unexpected many loops..");
            break;
        }
    } while (cur_n_size > 2);

#ifdef DEBUG_OUTPUT
    timer.reset();
    logger_->info("===== swap sort (2)");
#endif
    unsigned int num_m_blocks = get_num_blocks(m_, defaults::num_threads_in_swap_sort_func);

    swap_sort<<<num_m_blocks, defaults::num_threads_in_swap_sort_func>>>(
            n_stride,
            n_stride * m_,
            thrust::raw_pointer_cast(val.data()),
            thrust::raw_pointer_cast(idx.data()));
#ifdef DEBUG_OUTPUT
    logger_->info("swap sort (2) {}", timer.get_laptime());
    timer.reset();
#endif

    cudaMemcpy2D(thrust::raw_pointer_cast(&dom_data_->h_mins_val[0]), 2 * sizeof(double),
            thrust::raw_pointer_cast(val.data()), n_stride * sizeof(double), 2 * sizeof(double), m_,
            cudaMemcpyDeviceToHost);
#ifdef DEBUG_OUTPUT
    logger_->info("copy dev to host (all val) {}", timer.get_laptime());
    timer.reset();
#endif

    cudaMemcpy2D(thrust::raw_pointer_cast(&dom_data_->h_mins_idx[0]), 2 * sizeof(unsigned int),
            thrust::raw_pointer_cast(idx.data()), n_stride * sizeof(unsigned int), 2 * sizeof(unsigned int), m_,
            cudaMemcpyDeviceToHost);
#ifdef DEBUG_OUTPUT
    logger_->info("copy dev to host (all idx) {}", timer.get_laptime());
#endif

    cudaDeviceSynchronize();
}

bool NearestNeighborSearch::checkResult() {

#ifdef DEBUG_DIST_CHECK
    // check calculated distances
    double max_diff = 0.0;
    double sum_diff = 0.0;
    thrust::host_vector<double> r0(m_ * n_);
    for (unsigned int j = 0; j < n_; j++) {
        for (unsigned int i = 0; i < m_; i++) {
            double sum = 0.0;
            for (unsigned int w = 0; w < k_; w++) {
                double diff = dom_data_->h_x[i + w * m_] - dom_data_->h_y[j + w * n_];
                sum += diff * diff;
            }
            r0[j + i * n_] = sum;
            double diff_r = abs(dom_data_->h_r[j + i * n_] - sum);
            sum_diff += diff_r;
//            cout << "(i,j) " << i << "," << j << ": " << diff_r << " = " << r[j+i*n] << " - " << sum << endl;
            if (sum_diff > 0.01) {
                if (max_diff < diff_r) {
//                    cout << " MAX";
                    max_diff = diff_r;
                }
//                cout << endl;
            }
        }
    }
    logger_->info("avr diff: {}", sum_diff / (double)(m_ * n_));
    logger_->info("max diff: {}", max_diff);

    // check two nearest neighbor distances
    unsigned int num_incorrects = 0;
    if (sum_diff > 0.0)
        num_incorrects++;

    for (unsigned int i = 0; i < m_; i++) {
        double min_v = UINT_MAX;
        double next_v = UINT_MAX;
        unsigned int idx;
        for (unsigned int j = 0; j < n_; j++) {
            if (min_v > r0[j + i * n_]) {
                next_v = min_v;
                min_v = r0[j + i * n_];
                idx = j;
            } else if (next_v > r0[j + i * n_]) {
                next_v = r0[j + i * n_];
            }
        }
        if (min_v != dom_data_->h_mins_val[i * 2]) {
            num_incorrects++;
            logger_->warn("min value is diff. i={}: {}  -  {}", i, min_v, dom_data_->h_mins_val[i * 2]);
        }
        if (next_v != dom_data_->h_mins_val[1 + i * 2]) {
            num_incorrects++;
            logger_->warn("next min value is diff. i={}: {}  -  {}", i, next_v, dom_data_->h_mins_val[1 + i * 2]);
        }
        if (idx != dom_data_->h_mins_idx[i * 2]) {
            num_incorrects++;
            logger_->warn("index is diff. i={}: {}  -  {}", i, idx, dom_data_->h_mins_idx[i * 2]);
        }
    }
    logger_->info("# of incorrects={}", num_incorrects);
    if (num_incorrects > 0) {
        logger_->info("mins_val");
        print_matrix(2, 0, 0, 2, m_, dom_data_->h_mins_val);
        logger_->info("mins_idx");
        print_matrix(2, 0, 0, 2, m_, dom_data_->h_mins_idx);
        logger_->info("r (for check)");
        print_matrix(n_, 0, 0, n_, m_, dom_data_->h_r);
    }

    logger_->flush();

    return (num_incorrects == 0);
#else
    return true;
#endif
}

bool NearestNeighborSearch::checkDist2(double *dist2) {
    size_t count = 0;
#ifdef DEBUG_DIST_CHECK
    for (size_t i = 0; i < m_; i++) {
        for (size_t j = 0; j < n_; j++) {
            double diff = dist2[i + j * m_] - dom_data_->h_r[j + i * n_];
            if (fabs(diff) > 1e-10) {
                count++;
                logger_->error("i={},j={} diff {} vs {}", i, j, dist2[i + j * m_], dom_data_->h_r[j + i * n_]);
            }
            if (count > 100) {
                logger_->warn("too much diffs");
                return false;
            }
        }
    }
#endif

    return (count == 0);
}

} // namespace cudautils

