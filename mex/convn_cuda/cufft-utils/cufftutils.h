#ifndef __CUFFTUTILS_H__
#define __CUFFTUTILS_H__

#include <cufft.h>

namespace cufftutils {

    void get_pad_trim(int* size, int* filterdimA, int* pad_size, int trim_idxs[3][2]);

    long long convert_idx(long i, long j, long k, int* matrix_size, bool column_order);

    void convert_matrix(float* matrix, float* buffer, int* size, bool column_order);

    void initialize_inputs(float* hostI, float* hostF, cufftComplex* host_data_input, 
            cufftComplex* host_data_kernel, int* size, int* pad_size, int* filterdimA,
            bool column_order);

    int conv_handler(float* hostI, float* hostF, float* hostO, int algo, 
            int* dimA, int* filterdimA, bool column_order, int benchmark);

    int fft3(float * data, int* size, int* length, float* outArray, bool column_order);
}

#endif
