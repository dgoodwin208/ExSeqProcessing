#ifndef __NNSEARCH2_H__
#define __NNSEARCH2_H__

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <future>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include "cuda_task.h"

#include "spdlog/spdlog.h"

//#define DEBUG_OUTPUT
//#define DEBUG_TIMER_OUTPUT
//#define DEBUG_DIST_CHECK

namespace cudautils {

// pinned memory on host
typedef thrust::host_vector<double, thrust::system::cuda::experimental::pinned_allocator<double>> pinnedDblHostVector;
typedef thrust::host_vector<unsigned int, thrust::system::cuda::experimental::pinned_allocator<unsigned int>> pinnedUIntHostVector;

inline
unsigned int get_num_blocks(const unsigned int total_size, const unsigned int block_size) {
    return (total_size + block_size - 1) / block_size;
}

inline
unsigned int get_delta(const unsigned int total_size, const unsigned int index, const unsigned int delta) {
    return ((index + 1) * delta < total_size ? delta : total_size - index * delta);
}

__global__
void sum_squared(
        unsigned int w,
        unsigned int k,
        double *x,
        double *x2);

__global__
void calc_squared_norm(
        unsigned int m,
        unsigned int n,
        unsigned int k,
        double *x,
        double *y,
        double *x2,
        double *y2,
        double *r);

__global__
void calc_squared_norm2(
        unsigned int m,
        unsigned int n,
        unsigned int k,
        double *x,
        double *y,
        double *x2,
        double *y2,
        double *r);

__global__
void get_two_mins(
        unsigned int n,
        unsigned int idx_start,
        double *r0,
        double *r,
        unsigned int *idx);

__global__
void get_two_mins_with_index(
        const unsigned int stride,
        const unsigned int n,
        const unsigned int m,
        double *x,
        unsigned int *idx);

__global__
void gather_values_on_blocks(
        const unsigned int stride,
        const unsigned int n_size,
        const unsigned int block_size,
        const unsigned int m,
        double *val,
        unsigned int* idx);

__global__
void swap_sort(
        const unsigned int stride,
        const unsigned int total_size,
        double *val,
        unsigned int *idx);


class NearestNeighborSearch : public parallelutils::CudaTask {

    //   This class provides a function to calculate distances between all the combinations of two sets of vectors,
    //   and then find two pairs of minimum distances
    //
    //   Input:
    //       two sets of vectors that equal to matrices, x and y;
    //           the dimension of x is m rows and k columns
    //           the dimension of y is n rows and k columns
    //       these matrices are colomn-major matrices
    //
    //   Output:
    //       a list of two distances and their indices that show the nearest neighbor elements in y from x

    const unsigned int m_;
    const unsigned int n_;
    const unsigned int k_;
    const unsigned int dm_;
    const unsigned int dn_;

    const unsigned int num_gpus_;
    const unsigned int num_streams_;

    unsigned int num_dm_;  // (num_dm_-1) * dm_ < m_ <= num_dm_ * dm_
    unsigned int num_dn_;  // (num_dn_-1) * dn_ < n_ <= num_dn_ * dn_

    unsigned int n_blocks_in_two_mins_;

    struct DomainDataOnHost {
        pinnedDblHostVector h_x; // m-rows x k-columns
        pinnedDblHostVector h_y; // n-rows x k-columns
#ifdef DEBUG_DIST_CHECK
        pinnedDblHostVector h_r; // n-rows x m-columns
#endif

        pinnedDblHostVector h_mins_val;
        pinnedUIntHostVector h_mins_idx;

        DomainDataOnHost(const unsigned int m, const unsigned int n, const unsigned int k, const unsigned int num_dn)
        : h_x(m * k), h_y(n * k),
#ifdef DEBUG_DIST_CHECK
        h_r(size_t(n) * size_t(m)),
#endif
        h_mins_val(2 * num_dn * m), h_mins_idx(2 * num_dn * m)
        {
        }
    };

    std::shared_ptr<DomainDataOnHost> dom_data_;

    struct SubDomainDataOnStream {
        cudaStream_t stream;

        std::vector<unsigned int> x_i_list;

        thrust::device_vector<double> x;
        thrust::device_vector<double> x2;
        thrust::device_vector<double> r;

        thrust::device_vector<double> val;
        thrust::device_vector<unsigned int> idx;
    };

    struct SubDomainDataOnGPU {
        std::vector<unsigned int> y_i_list;

        thrust::device_vector<double> y;
        thrust::device_vector<double> y2;

        std::vector<std::shared_ptr<SubDomainDataOnStream>> stream_data;
    };

    std::vector<std::shared_ptr<SubDomainDataOnGPU>> subdom_data_;

    std::shared_ptr<spdlog::logger> logger_;

public:
    NearestNeighborSearch(
            const unsigned int m,
            const unsigned int n,
            const unsigned int k,
            const unsigned int dm,
            const unsigned int dn,
            const unsigned int num_gpus,
            const unsigned int num_streams);

    virtual ~NearestNeighborSearch();


    void generateSequences();
    void setInput(double* in_x, double* in_y);
    void setInput(std::vector<double>& in_x, std::vector<double>& in_y);
    void getResult(double* out_mins_val, unsigned int* out_mins_idx);
    void getResult(std::vector<double>& out_mins_val, std::vector<unsigned int>& out_mins_idx);
    void getDist2(double** out_dist2);
    double getDist2(size_t i, size_t j);

    virtual int getNumOfGPUTasks(const int gpu_id);
    virtual int getNumOfStreamTasks(const int gpu_id, const int stream_id);

    virtual void prerun() {}
    virtual void postrun();

    virtual void runOnGPU(const int gpu_id, const unsigned int gpu_task_id);
    virtual void postrunOnGPU(const int gpu_id, const unsigned int gpu_task_id) {}
    virtual void runOnStream( const int gpu_id, const int stream_id, const unsigned int gpu_task_id);

//    void run();
//    int runOnGPU(const unsigned int idx_gpu, const unsigned int task_id);
//    int runOnStream(const unsigned int idx_gpu, const unsigned int s_i, const unsigned int y_i);


    void precacheSquaredDistance(
            const unsigned int n,
            const unsigned int k,
            const unsigned int dn,
            const unsigned int y_start,
            pinnedDblHostVector& h_y,
            thrust::device_vector<double>& y,
            thrust::device_vector<double>& y2,
            cudaStream_t& stream);

    void calcSquaredDistanceWithCachedY(
            const unsigned int m, const unsigned int k,
            const unsigned int dm, const unsigned int dn,
            const unsigned int x_start, const unsigned int y_start,
            pinnedDblHostVector& h_x,
            thrust::device_vector<double>& x,
            thrust::device_vector<double>& x2,
            thrust::device_vector<double>& y,
            thrust::device_vector<double>& y2,
            thrust::device_vector<double>& r,
            cudaStream_t& stream);

    void getTwoTopsOfMinsInBlock(
            const unsigned int idx_gpu,
            const unsigned int s_i,
            const unsigned int x_i,
            const unsigned int y_i,
            const unsigned int m_steps,
            const unsigned int n_steps,
            const unsigned int y_start);

    void getTotalTwoTopsOfMins();


    bool checkResult();
    bool checkDist2(double *dist2);


    template <typename T>
    void print_matrix_common(
            const size_t rows,
            const size_t row_start,
            const size_t col_start,
            const size_t drows,
            const size_t dcols,
            T& val) {

        std::ostringstream sout;
        sout << "        ";
        for (size_t j = 0; j < dcols; j++) {
            sout << std::setw(6) << col_start + j << ",";
        }
        logger_->info("{}", sout.str());

        for (size_t i = 0; i < drows; i++) {
            sout.str("");
            sout.clear();
            sout << std::setw(6) << i << "| ";
            for (size_t j = 0; j < dcols; j++) {
                size_t idx = row_start + i + (col_start + j) * rows;
                sout << std::setw(6) << val[idx] << ",";
            }
            logger_->info("{}", sout.str());
        }
    }

    template <typename T>
    void print_matrix(
            const size_t rows,
            const size_t row_start,
            const size_t col_start,
            const size_t drows,
            const size_t dcols,
            thrust::device_vector<T>& val) {

        thrust::host_vector<T> h_val(val);
        print_matrix_common(rows, row_start, col_start, drows, dcols, h_val);
    }

    template <typename T>
    void print_matrix(
            const size_t rows,
            const size_t row_start,
            const size_t col_start,
            const size_t drows,
            const size_t dcols,
            thrust::host_vector<T>& val) {

        print_matrix_common(rows, row_start, col_start, drows, dcols, val);
    }

    template <typename T>
    void print_matrix(
            const size_t rows,
            const size_t row_start,
            const size_t col_start,
            const size_t drows,
            const size_t dcols,
            thrust::host_vector<T, thrust::system::cuda::experimental::pinned_allocator<T>>& val) {

        print_matrix_common(rows, row_start, col_start, drows, dcols, val);
    }

}; // class NearestNeighborSearch


} // namespace cudautils

#endif // __NNSEARCH2_H__

