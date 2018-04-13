#ifndef __CUFFTUTILS_H__
#define __CUFFTUTILS_H__

#include <cufft.h>

namespace cufftutils {

    void printHostData(cufftComplex *a, int size);

    void printDeviceData(cufftComplex *a, int size);

    void get_pad_trim(int* size, int* filterdimA, int* pad_size, int trim_idxs[3][2]);

    long long convert_idx(long i, long j, long k, int* matrix_size, bool column_order);

    void convert_matrix(float* matrix, float* buffer, int* size, bool column_order);

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
