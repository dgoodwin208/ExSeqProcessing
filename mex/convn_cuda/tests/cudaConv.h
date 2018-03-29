#ifndef CUDACONV_H
#define CUDACONV_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda.h>

namespace cuda3Dutils {

typedef enum signaltype {REAL, COMPLEX} signal;

typedef float2 Complex;


int
checkCUDA();

void cudaConvInit();

void
printHostData(Complex *a, int size, char *msg);


void
printDeviceData(Complex *a, int size, char *msg);

void
normData(Complex *a, int size, float norm);


void
randomFill(Complex *h_signal, int size, int flag);

void
zeroFill(Complex *h_signal, int size);

void
signalFFT1D(Complex *d_signal, int signal_size);


void
signalIFFT1D(Complex *d_signal, int signal_size);

void
signalFFT3D(Complex *d_signal, int NX, int NY, int NZ);

void signalIFFT3D(Complex *d_signal, int NX, int NY, int NZ);


__global__ void
pwProd(Complex *signal1, int size1, Complex *signal2);


void
cudaConvolution1D(Complex *d_signal1, int size1, Complex *d_signal2,
                int size2, dim3 blockSize, dim3 gridSize);

void
cudaConvolution3D(Complex *d_signal1, int* size1, Complex *d_signal2,
                int* size2, dim3 blockSize, dim3 gridSize);

int
allocateAndPad1D(Complex **a, int s1, Complex **b, int s2);

int
allocateAndPad3D(Complex **a, int s1, Complex **b, int s2);

} // namespace
#endif
