#include <cstdint>
#include <cassert>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/system/tbb/execution_policy.h>

#include "radixsort.h"

namespace cudautils {

template <typename T1, typename T2>
void radixsort(std::vector<T1>& array1, std::vector<T2>& array2)
{
    assert(array1.size() == array2.size());
    size_t array_size = array1.size();

    cudaError_t err;
    T1 *d_keys;
    T2 *d_values;
    err = cudaMalloc(&d_keys,   array_size * sizeof(T1));
    if (err == cudaErrorMemoryAllocation) {
        throw std::bad_alloc();
    }
    err= cudaMalloc(&d_values, array_size * sizeof(T2));
    if (err == cudaErrorMemoryAllocation) {
        throw std::bad_alloc();
    }

    cudaMemcpy(d_keys,   array1.data(), array_size * sizeof(T1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, array2.data(), array_size * sizeof(T2), cudaMemcpyHostToDevice);

    thrust::sort_by_key(thrust::device, d_keys, d_keys + array_size, d_values);

    cudaMemcpy(array1.data(), d_keys,   array_size * sizeof(T1), cudaMemcpyDeviceToHost);
    cudaMemcpy(array2.data(), d_values, array_size * sizeof(T2), cudaMemcpyDeviceToHost);
}

template
void radixsort<uint16_t, unsigned int>(std::vector<uint16_t>& array1, std::vector<unsigned int>& array2);

template
void radixsort<unsigned int, double>(std::vector<unsigned int>& array1, std::vector<double>& array2);

}

