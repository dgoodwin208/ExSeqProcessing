#ifndef __CUFFTUTILS_H__
#define __CUFFTUTILS_H__

#include <cufft.h>

namespace cufftutils {

    void printHostData(cufftComplex *a, int size);

    void printDeviceData(cufftComplex *a, int size);

    void get_pad_trim(int* size, int* filterdimA, int* pad_size, int trim_idxs[3][2]);

    __device__ __host__
    long long convert_idx(long i, long j, long k, int* matrix_size, bool column_order);

    void convert_matrix(float* matrix, float* buffer, int* size, bool column_order);

    __global__
    void complex_point_mul_scale_par(cufftComplex *a, cufftComplex *b, long long size, float scale);

    __global__
    void trim_pad_par(int trim_idxs00, int trim_idxs01, int trim_idxs10, 
            int trim_idxs11, int trim_idxs20, int trim_idxs21, int size0, int
            size1, int size2, int pad_size0, int pad_size1, int pad_size2, bool
            column_order, float* hostO, cufftComplex* host_data_input, bool
            benchmark) ;

    __global__
    void initialize_inputs_par(float* hostI, float* hostF, cufftComplex host_data_input[], 
            cufftComplex host_data_kernel[], int size0, int size1, int size2, int pad_size0, 
            int pad_size1, int pad_size2, int filterdimA0, int filterdimA1, int filterdimA2,
            bool column_order, int benchmark);

    void initialize_inputs(float* hostI, float* hostF, cufftComplex host_data_input[], 
            cufftComplex host_data_kernel[], int* size, int* pad_size, int* filterdimA,
            bool column_order);

    int conv_handler(float* hostI, float* hostF, float* hostO, int algo, 
            int* dimA, int* filterdimA, bool column_order, int benchmark);

    int conv_handler(float* hostI, float* hostF, float* hostO, int algo, 
            int* dimA, int* filterdimA, bool column_order, int benchmark);

    int conv_1GPU_handler(float* hostI, float* hostF, float* hostO, int algo, 
            int* dimA, int* filterdimA, bool column_order, int benchmark);

    int fft3(float * data, int* size, int* length, float* outArray, bool column_order);

    void product(cufftComplex *signal1, int size1, cufftComplex *signal2, dim3 gridSize, dim3 blockSize);

    void trim_pad(int trim_idxs[3][2], int* size, int* pad_size, bool column_order, float* hostO, cufftComplex* host_data_input) ;

    void signalFFT3D(cufftComplex *d_signal, int NX, int NY, int NZ);

    void signalIFFT3D(cufftComplex *d_signal, int NX, int NY, int NZ);

    void cudaConvolution3D(cufftComplex *d_signal1, int* size1, cufftComplex *d_signal2,
                    int* size2, dim3 blockSize, dim3 gridSize, int benchmark);


}

#endif
