#include "gtest/gtest.h"

#include "nnsearch2.h"
#include "defaults.h"

#include "spdlog/spdlog.h"

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <random>

namespace {

class NearestNeighborSearchCudaKernelTest : public ::testing::Test {
protected:
    std::shared_ptr<spdlog::logger> logger_;

    NearestNeighborSearchCudaKernelTest() {
        logger_ = spdlog::get("console");
        if (! logger_) {
            logger_ = spdlog::stdout_logger_mt("console");
        }
    }
    virtual ~NearestNeighborSearchCudaKernelTest() {
    }
    template <typename T>
    void print_matrix_debug(
            const size_t rows,
            const size_t row_start,
            const size_t col_start,
            const size_t drows,
            const size_t dcols,
            T& val) {

        std::cout << "        ";
        for (size_t j = 0; j < dcols; j++) {
            std::cout << std::setw(6) << col_start + j << ",";
        }
        std::cout << std::endl;

        for (size_t i = 0; i < drows; i++) {
            std::cout << std::setw(6) << i << "| ";
            for (size_t j = 0; j < dcols; j++) {
                size_t idx = row_start + i + (col_start + j) * rows;
                std::cout << std::setw(6) << val[idx] << ",";
            }
            std::cout << std::endl;
        }
    }

};

TEST_F(NearestNeighborSearchCudaKernelTest, SumSquared1Test) {

    unsigned int m = 6;
    unsigned int k = 3;

    thrust::device_vector<double> x(m * k);
    thrust::device_vector<double> x2(m);

    thrust::sequence(x.begin(), x.end());

    cudautils::sum_squared<<<1, 16>>>(m, k, thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(x2.data()));

    thrust::host_vector<double> h_x(x);
    thrust::host_vector<double> h_x2(x2);
    for (unsigned int i = 0; i < m; i++) {
        double sum = 0.0;
        for (unsigned int j = 0; j < k; j++) {
            sum += h_x[i + j * m] * h_x[i + j * m];
        }
        ASSERT_EQ(sum, h_x2[i]);
    }
}

TEST_F(NearestNeighborSearchCudaKernelTest, SumSquared2Test) {

    unsigned int m = 6;
    unsigned int k = 30;

    thrust::device_vector<double> x(m * k);
    thrust::device_vector<double> x2(m);

    thrust::sequence(x.begin(), x.end());

    cudautils::sum_squared<<<2, 16>>>(m, k, thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(x2.data()));

    thrust::host_vector<double> h_x(x);
    thrust::host_vector<double> h_x2(x2);
    for (unsigned int i = 0; i < m; i++) {
        double sum = 0.0;
        for (unsigned int j = 0; j < k; j++) {
            sum += h_x[i + j * m] * h_x[i + j * m];
        }
        ASSERT_EQ(sum, h_x2[i]);
    }
}

TEST_F(NearestNeighborSearchCudaKernelTest, CalcSquaredNorm1Test) {

    unsigned int m = 4;
    unsigned int n = 6;
    unsigned int k = 3;

    thrust::device_vector<double> x(m * k);
    thrust::device_vector<double> x2(m);
    thrust::device_vector<double> y(n * k);
    thrust::device_vector<double> y2(n);
    thrust::device_vector<double> r(m * n);

    thrust::sequence(x.begin(), x.end());
    thrust::sequence(y.begin(), y.end(), 1);

    unsigned int m_blocks = cudautils::get_num_blocks(m, 16);
    unsigned int n_blocks = cudautils::get_num_blocks(n, 16);
    cudautils::sum_squared<<<m_blocks, 16>>>(m, k, thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(x2.data()));
    cudautils::sum_squared<<<n_blocks, 16>>>(n, k, thrust::raw_pointer_cast(y.data()), thrust::raw_pointer_cast(y2.data()));

    m_blocks = cudautils::get_num_blocks(m, 32);
    n_blocks = cudautils::get_num_blocks(n, 32);
    dim3 dim_blocks(n_blocks, m_blocks, 1);
    dim3 dim_threads(32, 32, 1);
    cudautils::calc_squared_norm<<<dim_blocks, dim_threads>>>(
            m, n, k,
            thrust::raw_pointer_cast(x.data()),
            thrust::raw_pointer_cast(y.data()),
            thrust::raw_pointer_cast(x2.data()),
            thrust::raw_pointer_cast(y2.data()),
            thrust::raw_pointer_cast(r.data()));

    thrust::host_vector<double> h_x(x);
    thrust::host_vector<double> h_y(y);
    thrust::host_vector<double> h_r(r);
    for (unsigned int j = 0; j < n; j++) {
        for (unsigned int i = 0; i < m; i++) {
            double sum = 0.0;
            for (unsigned int w = 0; w < k; w++) {
                double diff = h_x[i + w * m] - h_y[j + w * n];
                sum += diff * diff;
            }
            ASSERT_EQ(sum, h_r[j + i * n]);
        }
    }
}

TEST_F(NearestNeighborSearchCudaKernelTest, CalcSquaredNorm2Test) {

    unsigned int m = 40;
    unsigned int n = 6;
    unsigned int k = 3;

    thrust::device_vector<double> x(m * k);
    thrust::device_vector<double> x2(m);
    thrust::device_vector<double> y(n * k);
    thrust::device_vector<double> y2(n);
    thrust::device_vector<double> r(m * n);

    thrust::sequence(x.begin(), x.end());
    thrust::sequence(y.begin(), y.end(), 1);

    unsigned int m_blocks = cudautils::get_num_blocks(m, 16);
    unsigned int n_blocks = cudautils::get_num_blocks(n, 16);
    cudautils::sum_squared<<<m_blocks, 16>>>(m, k, thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(x2.data()));
    cudautils::sum_squared<<<n_blocks, 16>>>(n, k, thrust::raw_pointer_cast(y.data()), thrust::raw_pointer_cast(y2.data()));

    m_blocks = cudautils::get_num_blocks(m, 32);
    n_blocks = cudautils::get_num_blocks(n, 32);
    dim3 dim_blocks(n_blocks, m_blocks, 1);
    dim3 dim_threads(32, 32, 1);
    cudautils::calc_squared_norm<<<dim_blocks, dim_threads>>>(
            m, n, k,
            thrust::raw_pointer_cast(x.data()),
            thrust::raw_pointer_cast(y.data()),
            thrust::raw_pointer_cast(x2.data()),
            thrust::raw_pointer_cast(y2.data()),
            thrust::raw_pointer_cast(r.data()));

    thrust::host_vector<double> h_x(x);
    thrust::host_vector<double> h_y(y);
    thrust::host_vector<double> h_r(r);
    for (unsigned int j = 0; j < n; j++) {
        for (unsigned int i = 0; i < m; i++) {
            double sum = 0.0;
            for (unsigned int w = 0; w < k; w++) {
                double diff = h_x[i + w * m] - h_y[j + w * n];
                sum += diff * diff;
            }
            ASSERT_EQ(sum, h_r[j + i * n]);
        }
    }
}

TEST_F(NearestNeighborSearchCudaKernelTest, CalcSquaredNorm3Test) {

    unsigned int m = 40;
    unsigned int n = 60;
    unsigned int k = 3;

    thrust::device_vector<double> x(m * k);
    thrust::device_vector<double> x2(m);
    thrust::device_vector<double> y(n * k);
    thrust::device_vector<double> y2(n);
    thrust::device_vector<double> r(m * n);

    thrust::sequence(x.begin(), x.end());
    thrust::sequence(y.begin(), y.end(), 1);

    unsigned int m_blocks = cudautils::get_num_blocks(m, 16);
    unsigned int n_blocks = cudautils::get_num_blocks(n, 16);
    cudautils::sum_squared<<<m_blocks, 16>>>(m, k, thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(x2.data()));
    cudautils::sum_squared<<<n_blocks, 16>>>(n, k, thrust::raw_pointer_cast(y.data()), thrust::raw_pointer_cast(y2.data()));

    m_blocks = cudautils::get_num_blocks(m, 32);
    n_blocks = cudautils::get_num_blocks(n, 32);
    dim3 dim_blocks(n_blocks, m_blocks, 1);
    dim3 dim_threads(32, 32, 1);
    cudautils::calc_squared_norm<<<dim_blocks, dim_threads>>>(
            m, n, k,
            thrust::raw_pointer_cast(x.data()),
            thrust::raw_pointer_cast(y.data()),
            thrust::raw_pointer_cast(x2.data()),
            thrust::raw_pointer_cast(y2.data()),
            thrust::raw_pointer_cast(r.data()));

    thrust::host_vector<double> h_x(x);
    thrust::host_vector<double> h_y(y);
    thrust::host_vector<double> h_r(r);
    for (unsigned int j = 0; j < n; j++) {
        for (unsigned int i = 0; i < m; i++) {
            double sum = 0.0;
            for (unsigned int w = 0; w < k; w++) {
                double diff = h_x[i + w * m] - h_y[j + w * n];
                sum += diff * diff;
            }
            ASSERT_EQ(sum, h_r[j + i * n]);
        }
    }
}

TEST_F(NearestNeighborSearchCudaKernelTest, GetTwoMins1Test) {

    // check if it is correct of choosing two minimum valus from a set of four values
    unsigned int m = 12;
    unsigned int n = 4;

    thrust::host_vector<double> h_r(m * n);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distr(0.2, 1.0);
    for (unsigned int i = 0; i < m * n; i++) {
        h_r[i] = distr(generator);
    }

    // [a, b, c, d] --> [a, b] < [c, d]

    // min: a, b; a < b  -> no movement
    h_r[0+0*n] = 0.10;//a
    h_r[1+0*n] = 0.15;//b

    // min: a, b; a > b  -> no movement
    h_r[0+1*n] = 0.15;//a
    h_r[1+1*n] = 0.10;//b

    // min: a, c; a < c
    h_r[0+2*n] = 0.10;//a
    h_r[2+2*n] = 0.15;//c

    // min: a, c; a > c
    h_r[0+3*n] = 0.15;//a
    h_r[2+3*n] = 0.10;//c

    // min: a, d; a < d
    h_r[0+4*n] = 0.10;//a
    h_r[3+4*n] = 0.15;//d

    // min: a, d; a > d
    h_r[0+5*n] = 0.15;//a
    h_r[3+5*n] = 0.10;//d

    // min: b, c; b < c
    h_r[1+6*n] = 0.10;//b
    h_r[2+6*n] = 0.15;//c

    // min: b, c; b > c
    h_r[1+7*n] = 0.15;//b
    h_r[2+7*n] = 0.10;//c

    // min: b, d; b < d
    h_r[1+8*n] = 0.10;//b
    h_r[3+8*n] = 0.15;//d

    // min: b, d; b > d
    h_r[1+9*n] = 0.15;//b
    h_r[3+9*n] = 0.10;//d

    // min: c, d; c < d
    h_r[2+10*n] = 0.10;//c
    h_r[3+10*n] = 0.15;//d

    // min: c, d; c > d
    h_r[2+11*n] = 0.15;//c
    h_r[3+11*n] = 0.10;//d

    unsigned int n_block_size = 2 * defaults::num_threads_in_twotops_func;
    unsigned int n_blocks = cudautils::get_num_blocks(n, n_block_size);
    unsigned int num_m_blocks_z = cudautils::get_num_blocks(m, defaults::num_blocks_y_in_twotops_func);
    unsigned int num_m_blocks_y = (num_m_blocks_z == 1) ? m : defaults::num_blocks_y_in_twotops_func;
    dim3 dim_blocks(n_blocks, num_m_blocks_y, num_m_blocks_z);

    thrust::device_vector<double> r(h_r);
    thrust::device_vector<double> val(2 * m);
    thrust::device_vector<unsigned int> idx(2 * m);

    cudautils::get_two_mins<<<dim_blocks, defaults::num_threads_in_twotops_func>>>(
        n, 0,
        thrust::raw_pointer_cast(r.data()),
        thrust::raw_pointer_cast(val.data()),
        thrust::raw_pointer_cast(idx.data()));

    thrust::host_vector<double> h_val(val);
    thrust::host_vector<double> h_idx(idx);
    ASSERT_EQ(0.10, h_val[0+0*2]);
    ASSERT_EQ(0.15, h_val[1+0*2]);

    ASSERT_EQ(0.15, h_val[0+1*2]);
    ASSERT_EQ(0.10, h_val[1+1*2]);

    ASSERT_EQ(0.10, h_val[0+2*2]);
    ASSERT_EQ(0.15, h_val[1+2*2]);

    ASSERT_EQ(0.10, h_val[0+3*2]);
    ASSERT_EQ(0.15, h_val[1+3*2]);

    ASSERT_EQ(0.10, h_val[0+4*2]);
    ASSERT_EQ(0.15, h_val[1+4*2]);

    ASSERT_EQ(0.15, h_val[0+5*2]);
    ASSERT_EQ(0.10, h_val[1+5*2]);

    ASSERT_EQ(0.15, h_val[0+6*2]);
    ASSERT_EQ(0.10, h_val[1+6*2]);

    ASSERT_EQ(0.10, h_val[0+7*2]);
    ASSERT_EQ(0.15, h_val[1+7*2]);

    ASSERT_EQ(0.15, h_val[0+8*2]);
    ASSERT_EQ(0.10, h_val[1+8*2]);

    ASSERT_EQ(0.15, h_val[0+9*2]);
    ASSERT_EQ(0.10, h_val[1+9*2]);

    ASSERT_EQ(0.10, h_val[0+10*2]);
    ASSERT_EQ(0.15, h_val[1+10*2]);

    ASSERT_EQ(0.15, h_val[0+11*2]);
    ASSERT_EQ(0.10, h_val[1+11*2]);


    ASSERT_EQ(0, h_idx[0+0*2]);
    ASSERT_EQ(1, h_idx[1+0*2]);

    ASSERT_EQ(0, h_idx[0+1*2]);
    ASSERT_EQ(1, h_idx[1+1*2]);

    ASSERT_EQ(0, h_idx[0+2*2]);
    ASSERT_EQ(2, h_idx[1+2*2]);

    ASSERT_EQ(2, h_idx[0+3*2]);
    ASSERT_EQ(0, h_idx[1+3*2]);

    ASSERT_EQ(0, h_idx[0+4*2]);
    ASSERT_EQ(3, h_idx[1+4*2]);

    ASSERT_EQ(0, h_idx[0+5*2]);
    ASSERT_EQ(3, h_idx[1+5*2]);

    ASSERT_EQ(2, h_idx[0+6*2]);
    ASSERT_EQ(1, h_idx[1+6*2]);

    ASSERT_EQ(2, h_idx[0+7*2]);
    ASSERT_EQ(1, h_idx[1+7*2]);

    ASSERT_EQ(3, h_idx[0+8*2]);
    ASSERT_EQ(1, h_idx[1+8*2]);

    ASSERT_EQ(1, h_idx[0+9*2]);
    ASSERT_EQ(3, h_idx[1+9*2]);

    ASSERT_EQ(2, h_idx[0+10*2]);
    ASSERT_EQ(3, h_idx[1+10*2]);

    ASSERT_EQ(2, h_idx[0+11*2]);
    ASSERT_EQ(3, h_idx[1+11*2]);

}

TEST_F(NearestNeighborSearchCudaKernelTest, GetTwoMins2Test) {

    // check if it is correct using n; 1*32 < n < 2*32 in a case of within one set of threads
    unsigned int n = 40;
    unsigned int m = n * (n - 1) / 2;

    thrust::host_vector<double> h_r(m * n);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distr(0.2, 1.0);
    for (unsigned int i = 0; i < m * n; i++) {
        h_r[i] = distr(generator);
    }

    unsigned int count = 0;
    for (unsigned int i = 0; i < n - 1; i++) {
        for (unsigned int j = i + 1; j < n; j++) {
            h_r[i+count*n] = 0.10;
            h_r[j+count*n] = 0.15;
            count++;
        }
    }

    unsigned int n_block_size = 2 * defaults::num_threads_in_twotops_func;
    unsigned int n_blocks = cudautils::get_num_blocks(n, n_block_size);
    unsigned int num_m_blocks_z = cudautils::get_num_blocks(m, defaults::num_blocks_y_in_twotops_func);
    unsigned int num_m_blocks_y = (num_m_blocks_z == 1) ? m : defaults::num_blocks_y_in_twotops_func;
    dim3 dim_blocks(n_blocks, num_m_blocks_y, num_m_blocks_z);

    thrust::device_vector<double> r(h_r);
    thrust::device_vector<double> val(2 * m);
    thrust::device_vector<unsigned int> idx(2 * m);

    cudautils::get_two_mins<<<dim_blocks, defaults::num_threads_in_twotops_func>>>(
        n, 0,
        thrust::raw_pointer_cast(r.data()),
        thrust::raw_pointer_cast(val.data()),
        thrust::raw_pointer_cast(idx.data()));

    thrust::host_vector<double> h_val(val);
    thrust::host_vector<double> h_idx(idx);
    count = 0;
    for (unsigned int i = 0; i < n - 1; i++) {
        for (unsigned int j = i + 1; j < n; j++) {
            double val1 = h_val[0+count*2];
            double val2 = h_val[1+count*2];
            unsigned int idx1 = h_idx[0+count*2];
            unsigned int idx2 = h_idx[1+count*2];

            ASSERT_TRUE(
                    (val1 == 0.10 && val2 == 0.15 && idx1 == i && idx2 == j) ||
                    (val2 == 0.10 && val1 == 0.15 && idx2 == i && idx1 == j));

            count++;
        }
    }
}

TEST_F(NearestNeighborSearchCudaKernelTest, GetTwoMins3Test) {

    // check if it is correct using n; 3*32 < n < 4*32 in a case of multi sets of threads but within one block
    unsigned int n = 120;
    unsigned int m = n * (n - 1) / 2;

    thrust::host_vector<double> h_r(m * n);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distr(0.2, 1.0);
    for (unsigned int i = 0; i < m * n; i++) {
        h_r[i] = distr(generator);
    }

    unsigned int count = 0;
    for (unsigned int i = 0; i < n - 1; i++) {
        for (unsigned int j = i + 1; j < n; j++) {
            h_r[i+count*n] = 0.10;
            h_r[j+count*n] = 0.15;
            count++;
        }
    }

    unsigned int n_block_size = 2 * defaults::num_threads_in_twotops_func;
    unsigned int n_blocks = cudautils::get_num_blocks(n, n_block_size);
    unsigned int num_m_blocks_z = cudautils::get_num_blocks(m, defaults::num_blocks_y_in_twotops_func);
    unsigned int num_m_blocks_y = (num_m_blocks_z == 1) ? m : defaults::num_blocks_y_in_twotops_func;
    dim3 dim_blocks(n_blocks, num_m_blocks_y, num_m_blocks_z);

    thrust::device_vector<double> r(h_r);
    thrust::device_vector<double> val(2 * m);
    thrust::device_vector<unsigned int> idx(2 * m);

    cudautils::get_two_mins<<<dim_blocks, defaults::num_threads_in_twotops_func>>>(
        n, 0,
        thrust::raw_pointer_cast(r.data()),
        thrust::raw_pointer_cast(val.data()),
        thrust::raw_pointer_cast(idx.data()));

    thrust::host_vector<double> h_val(val);
    thrust::host_vector<double> h_idx(idx);
    count = 0;
    for (unsigned int i = 0; i < n - 1; i++) {
        for (unsigned int j = i + 1; j < n; j++) {
            double val1 = h_val[0+count*2];
            double val2 = h_val[1+count*2];
            unsigned int idx1 = h_idx[0+count*2];
            unsigned int idx2 = h_idx[1+count*2];

            ASSERT_TRUE(
                    (val1 == 0.10 && val2 == 0.15 && idx1 == i && idx2 == j) ||
                    (val2 == 0.10 && val1 == 0.15 && idx2 == i && idx1 == j));

            count++;
        }
    }
}

TEST_F(NearestNeighborSearchCudaKernelTest, GetTwoMins4Test) {

    // check if it is correct using n > 2*32 in a case of multi blocks
    unsigned int n = 2000;
    unsigned int m = 2000;

    thrust::host_vector<double> h_r(m * n);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distr(0.2, 1.0);
    for (unsigned int i = 0; i < m * n; i++) {
        h_r[i] = distr(generator);
    }

    unsigned int count = 0;
    for (unsigned int i = 0; i < n - 1; i+=10) {
        for (unsigned int j = i + 1; j < n; j+=100) {
            h_r[i+count*n] = 0.10;
            h_r[j+count*n] = 0.15;

            count++;
            if (count == m) break;
        }
        if (count == m) break;
    }

    unsigned int n_block_size = 2 * defaults::num_threads_in_twotops_func;
    unsigned int n_blocks = cudautils::get_num_blocks(n, n_block_size);
    unsigned int num_m_blocks_z = cudautils::get_num_blocks(m, defaults::num_blocks_y_in_twotops_func);
    unsigned int num_m_blocks_y = (num_m_blocks_z == 1) ? m : defaults::num_blocks_y_in_twotops_func;
    dim3 dim_blocks(n_blocks, num_m_blocks_y, num_m_blocks_z);

    thrust::device_vector<double> r(h_r);
    thrust::device_vector<double> val(2 * n_blocks * m);
    thrust::device_vector<unsigned int> idx(2 * n_blocks * m);

    cudautils::get_two_mins<<<dim_blocks, defaults::num_threads_in_twotops_func>>>(
        n, 0,
        thrust::raw_pointer_cast(r.data()),
        thrust::raw_pointer_cast(val.data()),
        thrust::raw_pointer_cast(idx.data()));

    thrust::host_vector<double> h_val(val);
    thrust::host_vector<double> h_idx(idx);
    count = 0;
    for (unsigned int i = 0; i < n - 1; i+=10) {
        for (unsigned int j = i + 1; j < n; j+=100) {
            double val1 = std::numeric_limits<double>::max();
            double val2 = std::numeric_limits<double>::max();
            unsigned int idx1, idx2;
            for (unsigned int k = 0; k < n_blocks; k++) {
                double val_tmp1 = h_val[0+k*2+count*2*n_blocks];
                double val_tmp2 = h_val[1+k*2+count*2*n_blocks];
                unsigned int idx_tmp1 = h_idx[0+k*2+count*2*n_blocks];
                unsigned int idx_tmp2 = h_idx[1+k*2+count*2*n_blocks];

                if (val_tmp1 < val1) {
                    val2 = val1;
                    idx2 = idx1;
                    val1 = val_tmp1;
                    idx1 = idx_tmp1;
                } else if (val_tmp1 < val2) {
                    val2 = val_tmp1;
                    idx2 = idx_tmp1;
                }
                if (val_tmp2 < val1) {
                    val2 = val1;
                    idx2 = idx1;
                    val1 = val_tmp2;
                    idx1 = idx_tmp2;
                } else if (val_tmp2 < val2) {
                    val2 = val_tmp2;
                    idx2 = idx_tmp2;
                }
            }
            ASSERT_TRUE(
                    (val1 == 0.10 && val2 == 0.15 && idx1 == i && idx2 == j) ||
                    (val2 == 0.10 && val1 == 0.15 && idx2 == i && idx1 == j));

            count++;
            if (count == m) break;
        }
        if (count == m) break;
    }
}

TEST_F(NearestNeighborSearchCudaKernelTest, GetTwoMinsWithIndex1Test) {

    unsigned int m = 12;

    thrust::host_vector<double> h_val(4 * m);
    thrust::host_vector<unsigned int> h_idx(4 * m);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distr(0.2, 1.0);
    for (unsigned int i = 0; i < 4 * m; i++) {
        h_val[i] = distr(generator);
        h_idx[i] = 100;
    }

    // [a, b, c, d] --> [a, b] < [c, d]

    // min: a, b; a < b  -> no movement
    h_val[0+0*4] = 0.10;//a
    h_val[1+0*4] = 0.15;//b
    h_idx[0+0*4] = 0;
    h_idx[1+0*4] = 1;

    // min: a, b; a > b  -> no movement
    h_val[0+1*4] = 0.15;//a
    h_val[1+1*4] = 0.10;//b
    h_idx[0+1*4] = 0;
    h_idx[1+1*4] = 1;

    // min: a, c; a < c
    h_val[0+2*4] = 0.10;//a
    h_val[2+2*4] = 0.15;//c
    h_idx[0+2*4] = 0;
    h_idx[2+2*4] = 2;

    // min: a, c; a > c
    h_val[0+3*4] = 0.15;//a
    h_val[2+3*4] = 0.10;//c
    h_idx[0+3*4] = 0;
    h_idx[2+3*4] = 2;

    // min: a, d; a < d
    h_val[0+4*4] = 0.10;//a
    h_val[3+4*4] = 0.15;//d
    h_idx[0+4*4] = 0;
    h_idx[3+4*4] = 3;

    // min: a, d; a > d
    h_val[0+5*4] = 0.15;//a
    h_val[3+5*4] = 0.10;//d
    h_idx[0+5*4] = 0;
    h_idx[3+5*4] = 3;

    // min: b, c; b < c
    h_val[1+6*4] = 0.10;//b
    h_val[2+6*4] = 0.15;//c
    h_idx[1+6*4] = 1;
    h_idx[2+6*4] = 2;

    // min: b, c; b > c
    h_val[1+7*4] = 0.15;//b
    h_val[2+7*4] = 0.10;//c
    h_idx[1+7*4] = 1;
    h_idx[2+7*4] = 2;

    // min: b, d; b < d
    h_val[1+8*4] = 0.10;//b
    h_val[3+8*4] = 0.15;//d
    h_idx[1+8*4] = 1;
    h_idx[3+8*4] = 3;

    // min: b, d; b > d
    h_val[1+9*4] = 0.15;//b
    h_val[3+9*4] = 0.10;//d
    h_idx[1+9*4] = 1;
    h_idx[3+9*4] = 3;

    // min: c, d; c < d
    h_val[2+10*4] = 0.10;//c
    h_val[3+10*4] = 0.15;//d
    h_idx[2+10*4] = 2;
    h_idx[3+10*4] = 3;

    // min: c, d; c > d
    h_val[2+11*4] = 0.15;//c
    h_val[3+11*4] = 0.10;//d
    h_idx[2+11*4] = 2;
    h_idx[3+11*4] = 3;

    unsigned int n_blocks = 4;
    unsigned int num_m_blocks_z = cudautils::get_num_blocks(m, defaults::num_blocks_y_in_twotops_func);
    unsigned int num_m_blocks_y = (num_m_blocks_z == 1) ? m : defaults::num_blocks_y_in_twotops_func;
    dim3 dim_blocks(n_blocks, num_m_blocks_y, num_m_blocks_z);

    thrust::device_vector<double> val(h_val);
    thrust::device_vector<unsigned int> idx(h_idx);

    cudautils::get_two_mins_with_index<<<dim_blocks, defaults::num_threads_in_twotops_func>>>(
        4, n_blocks, m,
        thrust::raw_pointer_cast(val.data()),
        thrust::raw_pointer_cast(idx.data()));

    thrust::host_vector<double> h_ret_val(val);
    thrust::host_vector<unsigned int> h_ret_idx(idx);
    ASSERT_EQ(0.10, h_ret_val[0+0*4]);
    ASSERT_EQ(0.15, h_ret_val[1+0*4]);

    ASSERT_EQ(0.15, h_ret_val[0+1*4]);
    ASSERT_EQ(0.10, h_ret_val[1+1*4]);

    ASSERT_EQ(0.10, h_ret_val[0+2*4]);
    ASSERT_EQ(0.15, h_ret_val[1+2*4]);

    ASSERT_EQ(0.10, h_ret_val[0+3*4]);
    ASSERT_EQ(0.15, h_ret_val[1+3*4]);

    ASSERT_EQ(0.10, h_ret_val[0+4*4]);
    ASSERT_EQ(0.15, h_ret_val[1+4*4]);

    ASSERT_EQ(0.15, h_ret_val[0+5*4]);
    ASSERT_EQ(0.10, h_ret_val[1+5*4]);

    ASSERT_EQ(0.15, h_ret_val[0+6*4]);
    ASSERT_EQ(0.10, h_ret_val[1+6*4]);

    ASSERT_EQ(0.10, h_ret_val[0+7*4]);
    ASSERT_EQ(0.15, h_ret_val[1+7*4]);

    ASSERT_EQ(0.15, h_ret_val[0+8*4]);
    ASSERT_EQ(0.10, h_ret_val[1+8*4]);

    ASSERT_EQ(0.15, h_ret_val[0+9*4]);
    ASSERT_EQ(0.10, h_ret_val[1+9*4]);

    ASSERT_EQ(0.10, h_ret_val[0+10*4]);
    ASSERT_EQ(0.15, h_ret_val[1+10*4]);

    ASSERT_EQ(0.15, h_ret_val[0+11*4]);
    ASSERT_EQ(0.10, h_ret_val[1+11*4]);


    ASSERT_EQ(0, h_ret_idx[0+0*4]);
    ASSERT_EQ(1, h_ret_idx[1+0*4]);

    ASSERT_EQ(0, h_ret_idx[0+1*4]);
    ASSERT_EQ(1, h_ret_idx[1+1*4]);

    ASSERT_EQ(0, h_ret_idx[0+2*4]);
    ASSERT_EQ(2, h_ret_idx[1+2*4]);

    ASSERT_EQ(2, h_ret_idx[0+3*4]);
    ASSERT_EQ(0, h_ret_idx[1+3*4]);

    ASSERT_EQ(0, h_ret_idx[0+4*4]);
    ASSERT_EQ(3, h_ret_idx[1+4*4]);

    ASSERT_EQ(0, h_ret_idx[0+5*4]);
    ASSERT_EQ(3, h_ret_idx[1+5*4]);

    ASSERT_EQ(2, h_ret_idx[0+6*4]);
    ASSERT_EQ(1, h_ret_idx[1+6*4]);

    ASSERT_EQ(2, h_ret_idx[0+7*4]);
    ASSERT_EQ(1, h_ret_idx[1+7*4]);

    ASSERT_EQ(3, h_ret_idx[0+8*4]);
    ASSERT_EQ(1, h_ret_idx[1+8*4]);

    ASSERT_EQ(1, h_ret_idx[0+9*4]);
    ASSERT_EQ(3, h_ret_idx[1+9*4]);

    ASSERT_EQ(2, h_ret_idx[0+10*4]);
    ASSERT_EQ(3, h_ret_idx[1+10*4]);

    ASSERT_EQ(2, h_ret_idx[0+11*4]);
    ASSERT_EQ(3, h_ret_idx[1+11*4]);

}

TEST_F(NearestNeighborSearchCudaKernelTest, GetTwoMinsWithIndex2Test) {

    unsigned int n = 40;
    unsigned int m = n * (n - 1) / 2;

    thrust::host_vector<double> h_val(n * m);
    thrust::host_vector<unsigned int> h_idx(n * m);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distr(0.2, 1.0);
    for (unsigned int i = 0; i < n * m; i++) {
        h_val[i] = distr(generator);
        h_idx[i] = 100;
    }

    unsigned int count = 0;
    for (unsigned int i = 0; i < n - 1; i++) {
        for (unsigned int j = i + 1; j < n; j++) {
            h_val[i+count*n] = 0.10;
            h_val[j+count*n] = 0.15;
            h_idx[i+count*n] = i;
            h_idx[j+count*n] = j;
            count++;
        }
    }

    unsigned int n_block_size = 2 * defaults::num_threads_in_twotops_func;
    unsigned int n_blocks = cudautils::get_num_blocks(n, n_block_size);
    unsigned int num_m_blocks_z = cudautils::get_num_blocks(m, defaults::num_blocks_y_in_twotops_func);
    unsigned int num_m_blocks_y = (num_m_blocks_z == 1) ? m : defaults::num_blocks_y_in_twotops_func;
    dim3 dim_blocks(n_blocks, num_m_blocks_y, num_m_blocks_z);

    thrust::device_vector<double> val(h_val);
    thrust::device_vector<unsigned int> idx(h_idx);

    cudautils::get_two_mins_with_index<<<dim_blocks, defaults::num_threads_in_twotops_func>>>(
        n, n, m,
        thrust::raw_pointer_cast(val.data()),
        thrust::raw_pointer_cast(idx.data()));

    thrust::host_vector<double> h_ret_val(val);
    thrust::host_vector<unsigned int> h_ret_idx(idx);
    count = 0;
    for (unsigned int i = 0; i < n - 1; i++) {
        for (unsigned int j = i + 1; j < n; j++) {
            double val1 = h_ret_val[0+count*n];
            double val2 = h_ret_val[1+count*n];
            unsigned int idx1 = h_ret_idx[0+count*n];
            unsigned int idx2 = h_ret_idx[1+count*n];

            ASSERT_TRUE(
                    (val1 == 0.10 && val2 == 0.15 && idx1 == i && idx2 == j) ||
                    (val2 == 0.10 && val1 == 0.15 && idx2 == i && idx1 == j));

            count++;
        }
    }
}

TEST_F(NearestNeighborSearchCudaKernelTest, GetTwoMinsWithIndex3Test) {

    unsigned int n = 120;
    unsigned int m = n * (n - 1) / 2;

    thrust::host_vector<double> h_val(n * m);
    thrust::host_vector<unsigned int> h_idx(n * m);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distr(0.2, 1.0);
    for (unsigned int i = 0; i < n * m; i++) {
        h_val[i] = distr(generator);
        h_idx[i] = std::numeric_limits<unsigned int>::max();
    }

    unsigned int count = 0;
    for (unsigned int i = 0; i < n - 1; i++) {
        for (unsigned int j = i + 1; j < n; j++) {
            h_val[i+count*n] = 0.10;
            h_val[j+count*n] = 0.15;
            h_idx[i+count*n] = i;
            h_idx[j+count*n] = j;
            count++;
        }
    }

    unsigned int n_block_size = 2 * defaults::num_threads_in_twotops_func;
    unsigned int n_blocks = cudautils::get_num_blocks(n, n_block_size);
    unsigned int num_m_blocks_z = cudautils::get_num_blocks(m, defaults::num_blocks_y_in_twotops_func);
    unsigned int num_m_blocks_y = (num_m_blocks_z == 1) ? m : defaults::num_blocks_y_in_twotops_func;
    dim3 dim_blocks(n_blocks, num_m_blocks_y, num_m_blocks_z);

    thrust::device_vector<double> val(h_val);
    thrust::device_vector<unsigned int> idx(h_idx);

    cudautils::get_two_mins_with_index<<<dim_blocks, defaults::num_threads_in_twotops_func>>>(
        n, n, m,
        thrust::raw_pointer_cast(val.data()),
        thrust::raw_pointer_cast(idx.data()));

    thrust::host_vector<double> h_ret_val(val);
    thrust::host_vector<unsigned int> h_ret_idx(idx);
    count = 0;
    for (unsigned int i = 0; i < n - 1; i++) {
        for (unsigned int j = i + 1; j < n; j++) {
            double val1 = h_ret_val[0+count*n];
            double val2 = h_ret_val[1+count*n];
            unsigned int idx1 = h_ret_idx[0+count*n];
            unsigned int idx2 = h_ret_idx[1+count*n];

            ASSERT_TRUE(
                    (val1 == 0.10 && val2 == 0.15 && idx1 == i && idx2 == j) ||
                    (val2 == 0.10 && val1 == 0.15 && idx2 == i && idx1 == j));

            count++;
        }
    }
}

TEST_F(NearestNeighborSearchCudaKernelTest, GetTwoMinsWithIndex4Test) {

    unsigned int n = 2000;
    unsigned int m = 2000;

    thrust::host_vector<double> h_val(n * m);
    thrust::host_vector<unsigned int> h_idx(n * m);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distr(0.2, 1.0);
    for (unsigned int i = 0; i < n * m; i++) {
        h_val[i] = distr(generator);
        h_idx[i] = std::numeric_limits<unsigned int>::max();
    }

    unsigned int count = 0;
    for (unsigned int i = 0; i < n - 1; i+=10) {
        for (unsigned int j = i + 1; j < n; j+=100) {
            h_val[i+count*n] = 0.10;
            h_val[j+count*n] = 0.15;
            h_idx[i+count*n] = i;
            h_idx[j+count*n] = j;

            count++;
            if (count == m) break;
        }
        if (count == m) break;
    }

    unsigned int n_block_size = 2 * defaults::num_threads_in_twotops_func;
    unsigned int n_blocks = cudautils::get_num_blocks(n, n_block_size);
    unsigned int num_m_blocks_z = cudautils::get_num_blocks(m, defaults::num_blocks_y_in_twotops_func);
    unsigned int num_m_blocks_y = (num_m_blocks_z == 1) ? m : defaults::num_blocks_y_in_twotops_func;
    dim3 dim_blocks(n_blocks, num_m_blocks_y, num_m_blocks_z);

    thrust::device_vector<double> val(h_val);
    thrust::device_vector<unsigned int> idx(h_idx);

    cudautils::get_two_mins_with_index<<<dim_blocks, defaults::num_threads_in_twotops_func>>>(
        n, n, m,
        thrust::raw_pointer_cast(val.data()),
        thrust::raw_pointer_cast(idx.data()));

    thrust::host_vector<double> h_ret_val(val);
    thrust::host_vector<unsigned int> h_ret_idx(idx);
    count = 0;
    for (unsigned int i = 0; i < n - 1; i+=10) {
        for (unsigned int j = i + 1; j < n; j+=100) {
            double val1 = std::numeric_limits<double>::max();
            double val2 = std::numeric_limits<double>::max();
            unsigned int idx1, idx2;
            for (unsigned int k = 0; k < n_blocks; k++) {
                double val_tmp1 = h_ret_val[0+k*n_block_size+count*n];
                double val_tmp2 = h_ret_val[1+k*n_block_size+count*n];
                unsigned int idx_tmp1 = h_ret_idx[0+k*n_block_size+count*n];
                unsigned int idx_tmp2 = h_ret_idx[1+k*n_block_size+count*n];

                if (val_tmp1 < val1) {
                    val2 = val1;
                    idx2 = idx1;
                    val1 = val_tmp1;
                    idx1 = idx_tmp1;
                } else if (val_tmp1 < val2) {
                    val2 = val_tmp1;
                    idx2 = idx_tmp1;
                }
                if (val_tmp2 < val1) {
                    val2 = val1;
                    idx2 = idx1;
                    val1 = val_tmp2;
                    idx1 = idx_tmp2;
                } else if (val_tmp2 < val2) {
                    val2 = val_tmp2;
                    idx2 = idx_tmp2;
                }
            }
            ASSERT_TRUE(
                    (val1 == 0.10 && val2 == 0.15 && idx1 == i && idx2 == j) ||
                    (val2 == 0.10 && val1 == 0.15 && idx2 == i && idx1 == j));

            count++;
            if (count == m) break;
        }
        if (count == m) break;
    }
}

TEST_F(NearestNeighborSearchCudaKernelTest, GatherValuesOnBlocks1Test) {

    unsigned int m = 5;
    unsigned int stride = 12;
    unsigned int n_size = 12;
    unsigned int block_size = 2;

    thrust::host_vector<double> h_val(stride * m);
    thrust::host_vector<unsigned int> h_idx(stride * m);

    for (unsigned int i = 0; i < stride * m; i++) {
        h_val[i] = std::numeric_limits<double>::max();
        h_idx[i] = std::numeric_limits<unsigned int>::max();
    }

    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < n_size; j += block_size) {
            h_val[j     + i * stride] = 0.10;
            h_val[j + 1 + i * stride] = 0.15;
            h_idx[j     + i * stride] = j;
            h_idx[j + 1 + i * stride] = j + 1;
        }
    }

    thrust::device_vector<double> val(h_val);
    thrust::device_vector<unsigned int> idx(h_idx);

    cudautils::gather_values_on_blocks<<<m, 4>>>(
            stride, n_size, block_size, m,
            thrust::raw_pointer_cast(val.data()),
            thrust::raw_pointer_cast(idx.data()));

    thrust::host_vector<double> h_ret_val(val);
    thrust::host_vector<unsigned int> h_ret_idx(idx);

    for (unsigned int i = 0; i < m; i++) {
        unsigned int count = 0;
        for (unsigned int j = 0; j < n_size; j += block_size) {
            ASSERT_EQ(0.10,  h_ret_val[    2 * count + i * stride]);
            ASSERT_EQ(0.15,  h_ret_val[1 + 2 * count + i * stride]);
            ASSERT_EQ(j,     h_ret_idx[    2 * count + i * stride]);
            ASSERT_EQ(j + 1, h_ret_idx[1 + 2 * count + i * stride]);
            count++;
        }
    }
}

TEST_F(NearestNeighborSearchCudaKernelTest, GatherValuesOnBlocks2Test) {

    unsigned int m = 5;
    unsigned int stride = 12;
    unsigned int n_size = 12;
    unsigned int block_size = 4;

    thrust::host_vector<double> h_val(stride * m);
    thrust::host_vector<unsigned int> h_idx(stride * m);

    for (unsigned int i = 0; i < stride * m; i++) {
        h_val[i] = std::numeric_limits<double>::max();
        h_idx[i] = std::numeric_limits<unsigned int>::max();
    }

    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < n_size; j += block_size) {
            h_val[j     + i * stride] = 0.10;
            h_val[j + 1 + i * stride] = 0.15;
            h_idx[j     + i * stride] = j;
            h_idx[j + 1 + i * stride] = j + 1;
        }
    }

    thrust::device_vector<double> val(h_val);
    thrust::device_vector<unsigned int> idx(h_idx);

    cudautils::gather_values_on_blocks<<<m, 4>>>(
            stride, n_size, block_size, m,
            thrust::raw_pointer_cast(val.data()),
            thrust::raw_pointer_cast(idx.data()));

    thrust::host_vector<double> h_ret_val(val);
    thrust::host_vector<unsigned int> h_ret_idx(idx);

    for (unsigned int i = 0; i < m; i++) {
        unsigned int count = 0;
        for (unsigned int j = 0; j < n_size; j += block_size) {
            ASSERT_EQ(0.10,  h_ret_val[    2 * count + i * stride]);
            ASSERT_EQ(0.15,  h_ret_val[1 + 2 * count + i * stride]);
            ASSERT_EQ(j,     h_ret_idx[    2 * count + i * stride]);
            ASSERT_EQ(j + 1, h_ret_idx[1 + 2 * count + i * stride]);
            count++;
        }
    }
}

TEST_F(NearestNeighborSearchCudaKernelTest, GatherValuesOnBlocks3Test) {

    unsigned int n_block_size = 2 * defaults::num_threads_in_twotops_func;

    unsigned int m = 5;
    unsigned int stride = 200;
    unsigned int n_size = 120;
    unsigned int block_size = 40;

    unsigned int num_m_blocks_z = cudautils::get_num_blocks(m, defaults::num_blocks_y_in_twotops_func);
    unsigned int num_m_blocks_y = (num_m_blocks_z == 1) ? m : defaults::num_blocks_y_in_twotops_func;
    dim3 blocks_gather(num_m_blocks_y, num_m_blocks_z, 1);

    thrust::host_vector<double> h_val(stride * m);
    thrust::host_vector<unsigned int> h_idx(stride * m);

    for (unsigned int i = 0; i < stride * m; i++) {
        h_val[i] = std::numeric_limits<double>::max();
        h_idx[i] = std::numeric_limits<unsigned int>::max();
    }

    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < n_size; j += block_size) {
            h_val[j     + i * stride] = 0.10;
            h_val[j + 1 + i * stride] = 0.15;
            h_idx[j     + i * stride] = j;
            h_idx[j + 1 + i * stride] = j + 1;
        }
    }

    thrust::device_vector<double> val(h_val);
    thrust::device_vector<unsigned int> idx(h_idx);

    cudautils::gather_values_on_blocks<<<blocks_gather, n_block_size>>>(
            stride, n_size, block_size, m,
            thrust::raw_pointer_cast(val.data()),
            thrust::raw_pointer_cast(idx.data()));

    thrust::host_vector<double> h_ret_val(val);
    thrust::host_vector<unsigned int> h_ret_idx(idx);

    for (unsigned int i = 0; i < m; i++) {
        unsigned int count = 0;
        for (unsigned int j = 0; j < n_size; j += block_size) {
            ASSERT_EQ(0.10,  h_ret_val[    2 * count + i * stride]);
            ASSERT_EQ(0.15,  h_ret_val[1 + 2 * count + i * stride]);
            ASSERT_EQ(j,     h_ret_idx[    2 * count + i * stride]);
            ASSERT_EQ(j + 1, h_ret_idx[1 + 2 * count + i * stride]);
            count++;
        }
    }
}

TEST_F(NearestNeighborSearchCudaKernelTest, GatherValuesOnBlocks4Test) {

    unsigned int n_block_size = 2 * defaults::num_threads_in_twotops_func;

    unsigned int m = 5;
    unsigned int stride = 300;
    unsigned int n_size = 240;
    unsigned int block_size = 40;

    unsigned int num_m_blocks_z = cudautils::get_num_blocks(m, defaults::num_blocks_y_in_twotops_func);
    unsigned int num_m_blocks_y = (num_m_blocks_z == 1) ? m : defaults::num_blocks_y_in_twotops_func;
    dim3 blocks_gather(num_m_blocks_y, num_m_blocks_z, 1);

    thrust::host_vector<double> h_val(stride * m);
    thrust::host_vector<unsigned int> h_idx(stride * m);

    for (unsigned int i = 0; i < stride * m; i++) {
        h_val[i] = std::numeric_limits<double>::max();
        h_idx[i] = std::numeric_limits<unsigned int>::max();
    }

    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < n_size; j += block_size) {
            h_val[j     + i * stride] = 0.10;
            h_val[j + 1 + i * stride] = 0.15;
            h_idx[j     + i * stride] = j;
            h_idx[j + 1 + i * stride] = j + 1;
        }
    }

    thrust::device_vector<double> val(h_val);
    thrust::device_vector<unsigned int> idx(h_idx);

    cudautils::gather_values_on_blocks<<<blocks_gather, n_block_size>>>(
            stride, n_size, block_size, m,
            thrust::raw_pointer_cast(val.data()),
            thrust::raw_pointer_cast(idx.data()));

    thrust::host_vector<double> h_ret_val(val);
    thrust::host_vector<unsigned int> h_ret_idx(idx);

    for (unsigned int i = 0; i < m; i++) {
        unsigned int count = 0;
        for (unsigned int j = 0; j < n_size; j += block_size) {
            ASSERT_EQ(0.10,  h_ret_val[    2 * count + i * stride]);
            ASSERT_EQ(0.15,  h_ret_val[1 + 2 * count + i * stride]);
            ASSERT_EQ(j,     h_ret_idx[    2 * count + i * stride]);
            ASSERT_EQ(j + 1, h_ret_idx[1 + 2 * count + i * stride]);
            count++;
        }
    }
}

TEST_F(NearestNeighborSearchCudaKernelTest, GatherValuesOnBlocks5Test) {

    unsigned int n_block_size = 2 * defaults::num_threads_in_twotops_func;

    unsigned int m = 500;
    unsigned int stride = 300;
    unsigned int n_size = 240;
    unsigned int block_size = 40;

    unsigned int num_m_blocks_z = cudautils::get_num_blocks(m, defaults::num_blocks_y_in_twotops_func);
    unsigned int num_m_blocks_y = (num_m_blocks_z == 1) ? m : defaults::num_blocks_y_in_twotops_func;
    dim3 blocks_gather(num_m_blocks_y, num_m_blocks_z, 1);

    thrust::host_vector<double> h_val(stride * m);
    thrust::host_vector<unsigned int> h_idx(stride * m);

    for (unsigned int i = 0; i < stride * m; i++) {
        h_val[i] = std::numeric_limits<double>::max();
        h_idx[i] = std::numeric_limits<unsigned int>::max();
    }

    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < n_size; j += block_size) {
            h_val[j     + i * stride] = 0.10;
            h_val[j + 1 + i * stride] = 0.15;
            h_idx[j     + i * stride] = j;
            h_idx[j + 1 + i * stride] = j + 1;
        }
    }

    thrust::device_vector<double> val(h_val);
    thrust::device_vector<unsigned int> idx(h_idx);

    cudautils::gather_values_on_blocks<<<blocks_gather, n_block_size>>>(
            stride, n_size, block_size, m,
            thrust::raw_pointer_cast(val.data()),
            thrust::raw_pointer_cast(idx.data()));

    thrust::host_vector<double> h_ret_val(val);
    thrust::host_vector<unsigned int> h_ret_idx(idx);

    for (unsigned int i = 0; i < m; i++) {
        unsigned int count = 0;
        for (unsigned int j = 0; j < n_size; j += block_size) {
            ASSERT_EQ(0.10,  h_ret_val[    2 * count + i * stride]);
            ASSERT_EQ(0.15,  h_ret_val[1 + 2 * count + i * stride]);
            ASSERT_EQ(j,     h_ret_idx[    2 * count + i * stride]);
            ASSERT_EQ(j + 1, h_ret_idx[1 + 2 * count + i * stride]);
            count++;
        }
    }
}

TEST_F(NearestNeighborSearchCudaKernelTest, SwapSort1Test) {

    unsigned int stride = 10;
    unsigned int m = 5;
    unsigned int total_size = stride * m;

    thrust::host_vector<double> h_val(total_size);
    thrust::host_vector<unsigned int> h_idx(total_size);

    for (unsigned int i = 0; i < total_size; i++) {
        h_val[i] = std::numeric_limits<double>::max();
        h_idx[i] = std::numeric_limits<unsigned int>::max();
    }

    for (unsigned int i = 0; i < m; i++) {
        h_val[    i * stride] = 0.15;
        h_val[1 + i * stride] = 0.10;
        h_idx[    i * stride] = i;
        h_idx[1 + i * stride] = i + 1;
    }

    thrust::device_vector<double> val(h_val);
    thrust::device_vector<unsigned int> idx(h_idx);

    unsigned int num_blocks = cudautils::get_num_blocks(m, defaults::num_threads_in_swap_sort_func);
    cudautils::swap_sort<<<num_blocks, defaults::num_threads_in_swap_sort_func>>>(
            stride,
            total_size,
            thrust::raw_pointer_cast(val.data()),
            thrust::raw_pointer_cast(idx.data()));

    thrust::host_vector<double> h_ret_val(val);
    thrust::host_vector<unsigned int> h_ret_idx(idx);

    for (unsigned int i = 0; i < m; i++) {
        ASSERT_EQ(0.10,  h_ret_val[    i * stride]);
        ASSERT_EQ(0.15,  h_ret_val[1 + i * stride]);
        ASSERT_EQ(i + 1, h_ret_idx[    i * stride]);
        ASSERT_EQ(i,     h_ret_idx[1 + i * stride]);
    }
}

TEST_F(NearestNeighborSearchCudaKernelTest, SwapSort2Test) {

    unsigned int stride = 10;
    unsigned int m = 201;
    unsigned int total_size = stride * m;

    thrust::host_vector<double> h_val(total_size);
    thrust::host_vector<unsigned int> h_idx(total_size);

    for (unsigned int i = 0; i < total_size; i++) {
        h_val[i] = std::numeric_limits<double>::max();
        h_idx[i] = std::numeric_limits<unsigned int>::max();
    }

    for (unsigned int i = 0; i < m; i++) {
        h_val[    i * stride] = 0.15;
        h_val[1 + i * stride] = 0.10;
        h_idx[    i * stride] = i;
        h_idx[1 + i * stride] = i + 1;
    }

    thrust::device_vector<double> val(h_val);
    thrust::device_vector<unsigned int> idx(h_idx);

    unsigned int num_blocks = cudautils::get_num_blocks(m, defaults::num_threads_in_swap_sort_func);
    cudautils::swap_sort<<<num_blocks, defaults::num_threads_in_swap_sort_func>>>(
            stride,
            total_size,
            thrust::raw_pointer_cast(val.data()),
            thrust::raw_pointer_cast(idx.data()));

    thrust::host_vector<double> h_ret_val(val);
    thrust::host_vector<unsigned int> h_ret_idx(idx);

    for (unsigned int i = 0; i < m; i++) {
        ASSERT_EQ(0.10,  h_ret_val[    i * stride]);
        ASSERT_EQ(0.15,  h_ret_val[1 + i * stride]);
        ASSERT_EQ(i + 1, h_ret_idx[    i * stride]);
        ASSERT_EQ(i,     h_ret_idx[1 + i * stride]);
    }
}

TEST_F(NearestNeighborSearchCudaKernelTest, SwapSort3Test) {

    unsigned int stride = 1000;
    unsigned int m = 2000;
    unsigned int total_size = stride * m;

    thrust::host_vector<double> h_val(total_size);
    thrust::host_vector<unsigned int> h_idx(total_size);

    for (unsigned int i = 0; i < total_size; i++) {
        h_val[i] = std::numeric_limits<double>::max();
        h_idx[i] = std::numeric_limits<unsigned int>::max();
    }

    for (unsigned int i = 0; i < m; i++) {
        h_val[    i * stride] = 0.15;
        h_val[1 + i * stride] = 0.10;
        h_idx[    i * stride] = i;
        h_idx[1 + i * stride] = i + 1;
    }

    thrust::device_vector<double> val(h_val);
    thrust::device_vector<unsigned int> idx(h_idx);

    unsigned int num_blocks = cudautils::get_num_blocks(m, defaults::num_threads_in_swap_sort_func);
    cudautils::swap_sort<<<num_blocks, defaults::num_threads_in_swap_sort_func>>>(
            stride,
            total_size,
            thrust::raw_pointer_cast(val.data()),
            thrust::raw_pointer_cast(idx.data()));

    thrust::host_vector<double> h_ret_val(val);
    thrust::host_vector<unsigned int> h_ret_idx(idx);

    for (unsigned int i = 0; i < m; i++) {
        ASSERT_EQ(0.10,  h_ret_val[    i * stride]);
        ASSERT_EQ(0.15,  h_ret_val[1 + i * stride]);
        ASSERT_EQ(i + 1, h_ret_idx[    i * stride]);
        ASSERT_EQ(i,     h_ret_idx[1 + i * stride]);
    }
}

}

