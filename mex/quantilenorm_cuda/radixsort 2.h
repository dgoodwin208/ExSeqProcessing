#ifndef __RADIXSORT_H__
#define __RADIXSORT_H__

#include <vector>

namespace cudautils {

template <typename T1, typename T2>
void radixsort(std::vector<T1>& array1, std::vector<T2>& array2);

template <typename T1, typename T2>
void radixsort_host(std::vector<T1>& array1, std::vector<T2>& array2);

}

#endif // __RADIXSORT_H__

