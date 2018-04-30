#ifndef __MATRIX_HELPER_H__
#define __MATRIX_HELPER_H__

#include <memory>

#include "spdlog/spdlog.h"

namespace cudautils {

template <typename T>
void print_matrix_common(
        std::shared_ptr<spdlog::logger> logger,
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
    logger->info("{}", sout.str());

    for (size_t i = 0; i < drows; i++) {
        sout.str("");
        sout.clear();
        sout << std::setw(6) << i << "| ";
        for (size_t j = 0; j < dcols; j++) {
            size_t idx = row_start + i + (col_start + j) * rows;
            sout << std::setw(6);
            sout.operator<<(val[idx]);
            sout << ",";
        }
        logger->info("{}", sout.str());
    }
}

template <typename T>
void print_matrix(
        std::shared_ptr<spdlog::logger> logger,
        const size_t rows,
        const size_t row_start,
        const size_t col_start,
        const size_t drows,
        const size_t dcols,
        thrust::device_vector<T>& val) {

    thrust::host_vector<T> h_val(val);
    print_matrix_common(logger, rows, row_start, col_start, drows, dcols, h_val);
}

template <typename T>
void print_matrix(
        std::shared_ptr<spdlog::logger> logger,
        const size_t rows,
        const size_t row_start,
        const size_t col_start,
        const size_t drows,
        const size_t dcols,
        thrust::host_vector<T>& val) {

    print_matrix_common(logger, rows, row_start, col_start, drows, dcols, val);
}

template <typename T>
void print_matrix(
        std::shared_ptr<spdlog::logger> logger,
        const size_t rows,
        const size_t row_start,
        const size_t col_start,
        const size_t drows,
        const size_t dcols,
        thrust::host_vector<T, thrust::system::cuda::experimental::pinned_allocator<T>>& val) {

    print_matrix_common(logger, rows, row_start, col_start, drows, dcols, val);
}

template <typename T>
void print_matrix3d_common(
        std::shared_ptr<spdlog::logger> logger,
        const size_t x_size,
        const size_t y_size,
        const size_t x_start,
        const size_t y_start,
        const size_t z_start,
        const size_t dx,
        const size_t dy,
        const size_t dz,
        T& val) {

    std::ostringstream sout;
    sout << "        ";
    for (size_t j = 0; j < dy; j++) {
        sout << std::setw(6) << y_start + j << ",";
    }
    logger->debug("{}", sout.str());

    for (size_t k = 0; k < dz; k++) {
        logger->debug("[{}]", k);
        for (size_t i = 0; i < dx; i++) {
            sout.str("");
            sout.clear();
            sout << std::setw(6) << i << "| ";
            for (size_t j = 0; j < dy; j++) {
                size_t idx = x_start + i + (y_start + j) * x_size + (z_start + k) * x_size * y_size;
                sout << std::setw(6);
                sout.operator<<(val[idx]);
                sout << ",";
            }
            logger->debug("{}", sout.str());
        }
    }
}

template <typename T>
void print_matrix3d(
        std::shared_ptr<spdlog::logger> logger,
        const size_t x_size,
        const size_t y_size,
        const size_t x_start,
        const size_t y_start,
        const size_t z_start,
        const size_t dx,
        const size_t dy,
        const size_t dz,
        thrust::device_vector<T>& val) {

    thrust::host_vector<T> h_val(val);
    print_matrix3d_common(logger, x_size, y_size, x_start, y_start, z_start, dx, dy, dz, h_val);
}

template <typename T>
void print_matrix3d(
        std::shared_ptr<spdlog::logger> logger,
        const size_t x_size,
        const size_t y_size,
        const size_t x_start,
        const size_t y_start,
        const size_t z_start,
        const size_t dx,
        const size_t dy,
        const size_t dz,
        thrust::host_vector<T>& val) {

    print_matrix3d_common(logger, x_size, y_size, x_start, y_start, z_start, dx, dy, dz, val);
}

template <typename T>
void print_matrix3d(
        std::shared_ptr<spdlog::logger> logger,
        const size_t x_size,
        const size_t y_size,
        const size_t x_start,
        const size_t y_start,
        const size_t z_start,
        const size_t dx,
        const size_t dy,
        const size_t dz,
        thrust::host_vector<T, thrust::system::cuda::experimental::pinned_allocator<T>>& val) {

    print_matrix3d_common(logger, x_size, y_size, x_start, y_start, z_start, dx, dy, dz, val);
}


}

#endif // __MATRIX_HELPER_H__

