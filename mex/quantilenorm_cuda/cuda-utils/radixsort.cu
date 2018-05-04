#include <cstdint>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/system/tbb/execution_policy.h>

#include "radixsort.h"

namespace cudautils {

/*
 * Device code
 */



/*
 * Host code
 */
template <typename T1, typename T2>
void radixsort(std::vector<T1>& array1, std::vector<T2>& array2)
{
    thrust::device_vector<T1> d_keys(array1.begin(), array1.end());
    thrust::device_vector<T2> d_values(array2.begin(), array2.end());

    thrust::sort_by_key(thrust::device, d_keys.begin(), d_keys.end(), d_values.begin());

    thrust::copy(d_keys.begin(), d_keys.end(), array1.begin());
    thrust::copy(d_values.begin(), d_values.end(), array2.begin());
}

template
void radixsort<uint16_t, unsigned int>(std::vector<uint16_t>& array1, std::vector<unsigned int>& array2);

template
void radixsort<unsigned int, double>(std::vector<unsigned int>& array1, std::vector<double>& array2);

template <typename T1, typename T2>
void radixsort_host(std::vector<T1>& array1, std::vector<T2>& array2)
{
    thrust::host_vector<T1> h_keys(array1.begin(), array1.end());
    thrust::host_vector<T2> h_values(array2.begin(), array2.end());

    thrust::sort_by_key(thrust::tbb::par, h_keys.begin(), h_keys.end(), h_values.begin());

    thrust::copy(h_keys.begin(), h_keys.end(), array1.begin());
    thrust::copy(h_values.begin(), h_values.end(), array2.begin());
}

template
void radixsort_host<uint16_t, unsigned int>(std::vector<uint16_t>& array1, std::vector<unsigned int>& array2);

template
void radixsort_host<unsigned int, double>(std::vector<unsigned int>& array1, std::vector<double>& array2);

}

