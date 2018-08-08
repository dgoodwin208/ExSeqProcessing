// Multiple GPU version of cuFFT_check that uses multiple GPU's
// This program creates a real-valued 3D function sin(x)*cos(y)*cos(z) and then 
// takes the forward and inverse Fourier Transform, with the necessary scaling included. 
// The output of this process should match the input function

// includes, system
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <complex>

// includes, project
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime.h>

typedef thrust::host_vector<double, thrust::system::cuda::experimental::pinned_allocator<double>> pinnedDblHostVector;
typedef thrust::host_vector<unsigned int, thrust::system::cuda::experimental::pinned_allocator<int>> pinnedUIntHostVector;

namespace cufftutils {

// Pointwise Multiplication Kernel.
__global__ 
void pwProd(cufftComplex *signal1, int size1, cufftComplex *signal2) {

  int threadsPerBlock, blockId, globalIdx, i;
  cufftComplex temp;

  threadsPerBlock = blockDim.x * blockDim.y;
  blockId = blockIdx.x + (blockIdx.y * gridDim.x);
  globalIdx = (blockId * threadsPerBlock) + threadIdx.x + (threadIdx.y * blockDim.x);

  i = globalIdx;

  if (globalIdx < size1) {

    temp.x = (signal1[i].x * signal2[i].x) - (signal1[i].y * signal2[i].y);
    temp.y = (signal1[i].x * signal2[i].y) + (signal1[i].y * signal2[i].x);
    signal1[i].x = temp.x / (float) size1;
    signal1[i].y = temp.y / (float) size1;
  }

}


void printHostData(cufftComplex *a, int size) {

  for (int i = 0; i < size; i++)
    printf("%.1f %.1f\n", a[i].x, a[i].y);
}


void printDeviceData(cufftComplex *a, int size) {

  cufftComplex *h;

  h = (cufftComplex *) malloc(sizeof(cufftComplex) * size);

  cudaMemcpy(h, a, sizeof(cufftComplex) * size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < size; i++)
    printf("%f %f\n", h[i].x, h[i].y);

  free(h);
}


void product(cufftComplex *signal1, int size1, cufftComplex *signal2, dim3 gridSize, dim3 blockSize) {
    pwProd<<<gridSize, blockSize>>>(signal1, size1, signal2) ;
}

// 3D FFT on the _device_
void signalFFT3D(cufftComplex *d_signal, int NX, int NY, int NZ) {

  int NRANK, n[] = {NX, NY, NZ};
  cufftHandle plan;

  NRANK = 3;

    // Initializes the worksize variable
    /*size_t *worksize;                                   */
    // Allocates memory for the worksize variable, which tells cufft how many GPUs it has to work with
    /*worksize =(size_t*)malloc(sizeof(size_t) * 1);  */

  cufftResult result;
    //result = cufftCreate(&plan);
    //if (result != CUFFT_SUCCESS) { printf ("*Create plan failed\n"); exit(0); }
  //result = cufftMakePlan3d(plan, NX, NY, NZ, CUFFT_C2C, worksize); 
  result = cufftPlanMany(&plan, NRANK, n,
              NULL, 1, NX*NY*NZ, // *inembed, istride, idist
              NULL, 1, NX*NY*NZ, // *onembed, ostride, odist
              CUFFT_C2C, 1);
  if (result != CUFFT_SUCCESS) {
    printf ("Failed to plan 3D FFT code:%d\n", result);
    exit(0);
  }


  result = cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD) ; 
  if (result != CUFFT_SUCCESS){
    printf ("Failed to exec 3D FFT code %d\n", result);
    exit (0);
  }

}


// 3D IFFT on the _device_
void signalIFFT3D(cufftComplex *d_signal, int NX, int NY, int NZ) {

  int NRANK, n[] = {NX, NY, NZ};
  cufftHandle plan;

  NRANK = 3;

  if (cufftPlanMany(&plan, NRANK, n,
              NULL, 1, NX*NY*NZ, // *inembed, istride, idist
              NULL, 1, NX*NY*NZ, // *onembed, ostride, odist
              CUFFT_C2C, 1) != CUFFT_SUCCESS){
    printf ("Failed to plan 3D IFFT\n");
    exit (0);
  }


  if (cufftExecC2C(plan, d_signal, d_signal, CUFFT_INVERSE) != CUFFT_SUCCESS){
    printf ("Failed to exec 3D IFFT\n");
    exit (0);
  }

}

void cudaConvolution3D(cufftComplex *d_signal1, const unsigned int* size1, cufftComplex *d_signal2,
                const unsigned int* size2, dim3 blockSize, dim3 gridSize, int benchmark) {

    int N = size1[0] * size1[1] * size1[2];
    if (benchmark)
        printf("cudaConvolution3D");
    cufftutils::signalFFT3D(d_signal1, size1[0], size1[1], size1[2]);
    //if ((cudaGetLastError()) != cudaSuccess) {
    //printf ("signal 1 fft failed.\n");
    //exit(1);
    //}

    cufftutils::signalFFT3D(d_signal2, size2[0], size2[1], size2[2]);
    //if ((cudaGetLastError()) != cudaSuccess) {
    //printf ("signal 2 fft failed.\n");
    //exit(1);
    //}

    if (benchmark) {
        printf("\n Manual input FFT\n");
        cufftutils::printDeviceData(d_signal1, N);
        printf("\n Manual kernel FFT\n");
        cufftutils::printDeviceData(d_signal2, N);
    }

    cufftutils::product(d_signal1, N, d_signal2, gridSize, blockSize);
    //if ((cudaGetLastError()) != cudaSuccess) {
    //printf ("pwProd kernel failed.\n");
    //exit(1);
    //}
    //printDeviceData(d_signal1, size1, "PwProd");
    if (benchmark) {
        printf("\n Manual product \n");
        cufftutils::printDeviceData(d_signal1, N);
    }

    cufftutils::signalIFFT3D(d_signal1, size1[0], size1[1], size1[2]);
    //if ((cudaGetLastError()) != cudaSuccess) {
    //printf ("signal ifft failed.\n");
    //exit(1);
    //}
}

__global__ 
void initialize(int N, float* data, cufftComplex *f1, cufftComplex *f2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Initialize complex array from real input
        f1[idx].x = data[idx];
        f1[idx].y = 0.0; // complex component

        /*FIXME should this be real?*/
        // Initialize final array
        f2[idx].x = 0.0;
        f2[idx].y = 0.0;
    }

    return;
}

__global__
void scaleResult(int N, cufftComplex *f)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        f[idx].x = f[idx].x / ( (float)N );
        f[idx].y = f[idx].y / ( (float)N );
    }

    return;
}

// multiply, scale both arrays, keep product inplace on b
// caller must guarantee each point is reach via ample choice of gridSize, blockSize
__global__
void complex_point_mul_scale_par(cufftComplex *a, cufftComplex *b, long long size, float scale)
{
    long long i = ((long long) blockIdx.x) * blockDim.x + threadIdx.x;
    cufftComplex c;
    if (i < size) { // protect from overallocation of threads
        c = cuCmulf(a[i], b[i]);
        b[i] = make_cuFloatComplex(scale*cuCrealf(c), scale*cuCimagf(c) );
    }
    return;
}

// multiply, scale both arrays, keep product inplace on b
__global__
void ComplexPointwiseMulAndScale(cufftComplex *a, cufftComplex *b, int size, float scale)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    cufftComplex c;
    for (int i = threadID; i < size; i += numThreads)
    {
        c = cuCmulf(a[i], b[i]);
        b[i] = make_cuFloatComplex(scale*cuCrealf(c), scale*cuCimagf(c) );
    }
    return;
}

static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex a, cufftComplex b)
{
    cufftComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

static __device__ __host__ inline cufftComplex ComplexAdd(cufftComplex a, cufftComplex b)
{
    cufftComplex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}


void convolve_host(const cufftComplex *signal, long signal_size,
                      const cufftComplex *filter_kernel, long filter_kernel_size,
                                    cufftComplex *filtered_signal)
{
    long minRadius = filter_kernel_size / 2;
    long maxRadius = filter_kernel_size - minRadius;

    // Loop over output element indices
    for (int i = 0; i < signal_size; ++i)
    {
        filtered_signal[i].x = filtered_signal[i].y = 0;

        // Loop over convolution indices
        for (int j = - maxRadius + 1; j <= minRadius; ++j)
        {
            int k = i + j;

            if (k >= 0 && k < signal_size)
            {
                filtered_signal[i] = ComplexAdd(filtered_signal[i], ComplexMul(signal[k], filter_kernel[minRadius - j]));
            }
        }
    }
}


// Compute pad length per given dimension lengths
long get_pad_idx(long m, long n) {
    return m + n - 1;
}

__device__ __host__
long long convert_idx(long i, long j, long k, const unsigned int* matrix_size, bool column_order) {
    if (column_order) {
        return i + j * matrix_size[0] + ((long long) k) * matrix_size[0] * matrix_size[1];
    } else {
        return k + j * matrix_size[2] + ((long long) i) * matrix_size[2] * matrix_size[1];
    }
}

// converts from column order to c-order when column_to_c != 0, otherwise reversed
void convert_matrix(float* matrix, float* buffer, const unsigned int* size, bool column_order) {
    long long from_idx;
    long long to_idx;
    for ( long i = 0; i < size[0]; i++) { 
        for (long j = 0; j < size[1]; j++) {
            for (long k = 0; k < size[2]; k++) {

                from_idx = convert_idx(i, j, k, size, column_order);
                to_idx = convert_idx(i, j, k, size, !column_order);

                buffer[to_idx] = matrix[from_idx];
            }
        }
    }
}

__global__
void initialize_inputs_par(float* hostI, float* hostF, cufftComplex
        host_data_input[], cufftComplex host_data_kernel[], const unsigned int
        size0, const unsigned int size1, const unsigned int size2, const
        unsigned int pad_size0, const unsigned int pad_size1, const unsigned
        int pad_size2, const unsigned int filterdimA0, const unsigned int
        filterdimA1, const unsigned int filterdimA2, bool column_order, int
        benchmark) {
    
    // Place in matrix padded to 0
    const unsigned int pad_size[3] = {pad_size0, pad_size1, pad_size2};
    const unsigned int filterdimA[3] = {filterdimA0, filterdimA1, filterdimA2};
    const unsigned int size[3] = {size0, size1, size2};

    // identify corresponding index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    long long idx;
    long long idx_filter;
    long long pad_idx;
    /*for ( long i = index; i < pad_size[0]; i += stride) { */
    if ((i < pad_size[0]) && (j < pad_size[1]) && ( k < pad_size[2])) { 

        idx = convert_idx(i, j, k, size, column_order);
        idx_filter = convert_idx(i, j, k, filterdimA, column_order);
        // always place into c-order for cuda processing, revert in trim_pad()
        pad_idx = convert_idx(i, j, k, pad_size, false); 
        if (benchmark)
            printf("asdf %d,%d,%d idx:%d, idx_filter:%d, pad_idx:%d, hostI[idx]:%f\n", 
                    i, j, k, idx, idx_filter, pad_idx, hostI[idx]);

        if ((i < filterdimA[0]) && (j < filterdimA[1]) && (k < filterdimA[2])) {
            host_data_kernel[pad_idx].x = hostF[idx_filter];
        } else {
            host_data_kernel[pad_idx].x = 0.0f;
        }
        host_data_kernel[pad_idx].y = 0.0f; // y is complex component

        // keep in Matlab Column-order but switch order of dimensions in createPlan
        // to accomplish c-order FFT transforms
        if ((i < size[0]) && (j < size[1]) && (k < size[2]) ) {
            host_data_input[pad_idx].x = hostI[idx];
        } else {
            host_data_input[pad_idx].x = 0.0f;
        }
        host_data_input[pad_idx].y = 0.0f; 
    }
}

void initialize_inputs(float* hostI, float* hostF, cufftComplex host_data_input[], 
        cufftComplex host_data_kernel[], const unsigned int* size, unsigned int* pad_size, const unsigned int* filterdimA,
        bool column_order) {
    // Place in matrix padded to 0
    long long idx;
    long long idx_filter;
    long long pad_idx;
    for ( long i = 0; i < pad_size[0]; i += 1) { 
        for (long j = 0; j < pad_size[1]; j++) {
            for (long k = 0; k < pad_size[2]; k++) {

                idx = convert_idx(i, j, k, size, column_order);
                idx_filter = convert_idx(i, j, k, filterdimA, column_order);
                // always place into c-order for cuda processing, revert in trim_pad()
                pad_idx = convert_idx(i, j, k, pad_size, false); 

                if ((i < filterdimA[0]) && (j < filterdimA[1]) && (k < filterdimA[2])) {
                    host_data_kernel[pad_idx].x = hostF[idx_filter];
                } else {
                    host_data_kernel[pad_idx].x = 0.0f;
                }
                host_data_kernel[pad_idx].y = 0.0f; // y is complex component

                // keep in Matlab Column-order but switch order of dimensions in createPlan
                // to accomplish c-order FFT transforms
                if ((i < size[0]) && (j < size[1]) && (k < size[2]) ) {
                    host_data_input[pad_idx].x = hostI[idx];
                } else {
                    host_data_input[pad_idx].x = 0.0f;
                }
                host_data_input[pad_idx].y = 0.0f; 
            }
        }
    }
}

void get_pad_trim(const unsigned int* size, const unsigned int* filterdimA, unsigned int* pad_size, int trim_idxs[3][2]) {
    // Compute pad lengths
    for (int i=0; i < 3; i++) 
        pad_size[i] = cufftutils::get_pad_idx(size[i], filterdimA[i]);

    for (int i=0; i < 3; i++) {
        trim_idxs[i][0] = ceil((filterdimA[i] - 1) / 2);
        trim_idxs[i][1] = size[i] + ceil((filterdimA[i] - 1) / 2);
    }
}

__global__
void trim_pad_par(int trim_idxs00, int trim_idxs01, int trim_idxs10, 
        int trim_idxs11, int trim_idxs20, int trim_idxs21, const unsigned int size0, const unsigned int
        size1, const unsigned int size2, const unsigned int pad_size0, const unsigned int pad_size1, const unsigned int pad_size2, bool
        column_order, float* hostO, cufftComplex* host_data_input, bool
        benchmark) 
{
    // identify corresponding index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    const unsigned int pad_size[3] = {pad_size0, pad_size1, pad_size2};
    const unsigned int size[3] = {size0, size1, size2};

    long long idx;
    long long pad_idx;
    // FIXME check these ifs are legal in C
    if (trim_idxs00 <= i < trim_idxs01) {
        if (trim_idxs10 <= j < trim_idxs11) {
            if (trim_idxs20 <= k < trim_idxs21) {
                idx = cufftutils::convert_idx(i - trim_idxs00,
                        j - trim_idxs10, k - trim_idxs20, size, column_order);
                // data always processed, stored in c-order, see initialize_inputs()
                pad_idx = cufftutils::convert_idx(i, j, k, pad_size, false);

                hostO[idx] = host_data_input[pad_idx].x;
                if (benchmark)
                    printf("%f\n", host_data_input[pad_idx].x);
            }
        }
    }
}

void trim_pad(int trim_idxs[3][2], const unsigned int* size, const unsigned int* pad_size, bool column_order,
        float* hostO, cufftComplex* host_data_input, bool benchmark) 
{
    long long idx;
    long long pad_idx;
    for (long i=trim_idxs[0][0]; i < trim_idxs[0][1]; i++) {
        for (long j=trim_idxs[1][0]; j < trim_idxs[1][1]; j++) {
            for (long k=trim_idxs[2][0]; k < trim_idxs[2][1]; k++) {
                idx = cufftutils::convert_idx(i - trim_idxs[0][0],
                        j - trim_idxs[1][0], k - trim_idxs[2][0], size, column_order);
                // data always processed, stored in c-order, see initialize_inputs()
                pad_idx = cufftutils::convert_idx(i, j, k, pad_size, false);

                hostO[idx] = host_data_input[pad_idx].x;
                if (benchmark)
                    printf("%f\n", host_data_input[pad_idx].x);
            }
        }
    }
}

int conv_1GPU_handler(float* hostI, float* hostF, float* hostO, int algo, const unsigned int* size, const unsigned int* filterdimA, bool column_order, int benchmark) {
    // hardcoded, func only supports 3D convolutions

    long long N = ((long long) size[0]) * size[1] * size[2];
    if (benchmark)
        printf("Using %d GPUs on a %dx%dx%d grid, N:%d\n",1, size[0], size[1], size[2], N);

    unsigned int pad_size[3];
    int trim_idxs[3][2];
    cufftutils::get_pad_trim(size, filterdimA, pad_size, trim_idxs);

    long long N_padded = pad_size[0] * pad_size[1] * pad_size[2];
    long long size_of_data = N_padded * sizeof(cufftComplex);

    if (benchmark)
        printf("Padded to a %dx%dx%d grid, N:%d\n",pad_size[0], pad_size[1], pad_size[2], N_padded);

    //Create complex variables on host
    if (benchmark)
        printf("malloc input and output\n");
    cufftComplex *host_data_input = (cufftComplex *)malloc(size_of_data);
    if (!host_data_input) { printf("malloc input failed"); }
    cufftComplex *host_data_kernel = (cufftComplex *)malloc(size_of_data);
    if (!host_data_kernel) { printf("malloc kernel failed"); }

    float elapsed = 0.0f;
    cudaEvent_t start, stop;
    if (benchmark) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    if (benchmark)
        printf("Initialize input and kernel\n");

    cufftutils::initialize_inputs(hostI, hostF, host_data_input, host_data_kernel, size, pad_size, filterdimA, column_order);

    // Synchronize GPU before moving forward
    cudaDeviceSynchronize();

    if (benchmark) {
        printf("\n1GPU host_data_input elements:%d\n", N_padded);
        cufftutils::printDeviceData(host_data_input, N_padded);
        printf("\n1GPU host_data_kernel elements:%d\n", N_padded);
        cufftutils::printDeviceData(host_data_kernel, N_padded);
    }

    if (benchmark)
        printf("Input and output successfully initialized\n");

    if (benchmark)
        printf("cudaMalloc\n");
    cufftComplex* device_data_input;
    cufftComplex* device_data_kernel;
    cudaMalloc(&device_data_input, size_of_data);
    cudaMalloc(&device_data_kernel, size_of_data);

    if (benchmark)
        printf("cudaMemcpy\n");
    // Copy the data from 'host' to device using cufftXt formatting
    cudaMemcpy(device_data_input, host_data_input, size_of_data, cudaMemcpyHostToDevice);
    cudaMemcpy(device_data_kernel, host_data_kernel, size_of_data, cudaMemcpyHostToDevice);

    if (benchmark)
        printf("conv3D\n");
    const dim3 blockSize(16, 16, 1);
    const dim3 gridSize(N_padded / 16 + 1, N_padded / 16 + 1, 1);
    cufftutils::cudaConvolution3D(device_data_input, pad_size, device_data_kernel, pad_size, 
            blockSize, gridSize, benchmark);

    cudaMemcpy(host_data_input, device_data_input, size_of_data, cudaMemcpyDeviceToHost);

    if (benchmark) {
        printf("Print hostO_1GPU final\n");
        cufftutils::printHostData(host_data_input, N_padded);
    }

    cufftutils::trim_pad(trim_idxs, size, pad_size, column_order, hostO, host_data_input, benchmark);

    if (benchmark) {
        cudaEventRecord(stop, 0);
        cudaDeviceSynchronize();
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        printf(" Full convolution cost %.2f ms\n", elapsed);
    }

    if (benchmark)
        printf("Cufft completed successfully\n");

    // Free malloc'ed variables
    free(host_data_input);
    free(host_data_kernel);

    // Free cufftX malloc'ed variables
    cudaFree(device_data_input);
    cudaFree(device_data_kernel);

    return 0;
}

int conv_handler(float* hostI, float* hostF, float* hostO, int algo, const unsigned int* size, const unsigned int* filterdimA, bool column_order, int benchmark) {
    // hardcoded, func only supports 3D convolutions
    int nGPUs;
    cudaGetDeviceCount(&nGPUs);

    long long N = ((long long) size[0]) * size[1] * size[2];
    if (benchmark)
        printf("Using %d GPUs on a %dx%dx%d grid, N:%d\n",nGPUs, size[0], size[1], size[2], N);
    /*long long N_kernel = ((long long) filterdimA[0]) * filterdimA[1] * filterdimA[2];*/

    unsigned int pad_size[3];
    int trim_idxs[3][2];
    cufftutils::get_pad_trim(size, filterdimA, pad_size, trim_idxs);

    long long N_padded = pad_size[0] * pad_size[1] * pad_size[2];
    long long size_of_data = N_padded * sizeof(cufftComplex);

    if (benchmark)
        printf("Padded to a %dx%dx%d grid, N:%d\n",pad_size[0], pad_size[1], pad_size[2], N_padded);

    //Create complex variables on host
    if (benchmark)
        printf("malloc input and output\n");
    cufftComplex *host_data_input = (cufftComplex *)malloc(size_of_data);
    if (!host_data_input) { printf("malloc input failed"); }
    cufftComplex *host_data_kernel = (cufftComplex *)malloc(size_of_data);
    if (!host_data_kernel) { printf("malloc kernel failed"); }

    float elapsed = 0.0f;
    cudaEvent_t start, stop;
    if (benchmark) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    if (benchmark)
        printf("Initialize input and kernel\n");

    cufftutils::initialize_inputs(hostI, hostF, host_data_input, host_data_kernel, size, pad_size, filterdimA, column_order);

    /*// Synchronize GPUs before moving forward*/
    /*for (i = 0; i<nGPUs; ++i){*/
        /*cudaSetDevice(deviceNum[i]);*/
        /*cudaDeviceSynchronize();*/
    /*}*/

    if (benchmark) {
        /*printf("\nhost_data_input elements:%d\n", N_padded);*/
        /*cufftutils::printHostData(host_data_input, N_padded);*/
        /*printf("\nhost_data_kernel elements:%d\n", N_padded);*/
        /*cufftutils::printHostData(host_data_kernel, N_padded);*/
    }

    if (benchmark)
        printf("Input and output successfully initialized\n");

    // Set GPU's to use 
    int deviceNum[nGPUs];
    for(int i = 0; i<nGPUs; ++i)
    {
        deviceNum[i] = i;
    }

    // Initialize result variable used for error checking
    cufftResult result;

    // Create empty plan that will be used for FFT / IFFT
    cufftHandle plan_fft3;
    result = cufftCreate(&plan_fft3);
    if (result != CUFFT_SUCCESS) { printf ("*Create plan failed\n"); return 1; }

    // Tell cuFFT which GPUs to use
    result = cufftXtSetGPUs (plan_fft3, nGPUs, deviceNum);
    if (result != CUFFT_SUCCESS) { printf ("*XtSetGPUs failed: code %i\n", result); return 1; }

    // Initializes the worksize variable
    size_t *worksize;                                   
    // Allocates memory for the worksize variable, which tells cufft how many GPUs it has to work with
    worksize =(size_t*)malloc(sizeof(size_t) * nGPUs);  
    
    if (benchmark)
        printf("Make plan 3d\n");
    // Create the plan for cufft, each element of worksize is the workspace for that GPU
    // multi-gpus must have a complex to complex transform
    result = cufftMakePlan3d(plan_fft3, pad_size[0], pad_size[1], pad_size[2], CUFFT_C2C, worksize); 
    if (result != CUFFT_SUCCESS) { printf ("*MakePlan3d* failed: code %d \n",(int)result); exit (EXIT_FAILURE) ; }

    // Allocate data on multiple gpus using the cufft routines
    // Initialize transform array - to be split among GPU's and transformed in place using cufftX
    if (benchmark)
        printf("Allocate mGPU\n");
    cudaLibXtDesc *device_data_input, *device_data_kernel;
    result = cufftXtMalloc(plan_fft3, &device_data_input, CUFFT_XT_FORMAT_INPLACE);
    if (result != CUFFT_SUCCESS) { printf ("*XtMalloc failed, code: %d\n", result); exit (EXIT_FAILURE) ; }
    result = cufftXtMalloc(plan_fft3, &device_data_kernel, CUFFT_XT_FORMAT_INPLACE);
    if (result != CUFFT_SUCCESS) { printf ("*XtMalloc kernel failed, code: %d\n", result); exit (EXIT_FAILURE) ; }

    if (benchmark)
        printf("Xt memcpy\n");
    // Copy the data from 'host' to device using cufftXt formatting
    result = cufftXtMemcpy(plan_fft3, device_data_input, host_data_input, CUFFT_COPY_HOST_TO_DEVICE);
    if (result != CUFFT_SUCCESS) { printf ("*XtMemcpy failed, code: %d\n",result); exit (EXIT_FAILURE); }
    result = cufftXtMemcpy(plan_fft3, device_data_kernel, host_data_kernel, CUFFT_COPY_HOST_TO_DEVICE);
    if (result != CUFFT_SUCCESS) { printf ("*XtMemcpy failed, code: %d\n",result); exit (EXIT_FAILURE); }

    // Perform FFT on multiple GPUs
    if (benchmark)
        printf("Forward 3d FFT input on multiple GPUs\n");
    result = cufftXtExecDescriptorC2C(plan_fft3, device_data_input, device_data_input, CUFFT_FORWARD);
    if (result != CUFFT_SUCCESS) { printf ("*XtExecC2C input failed\n"); exit (EXIT_FAILURE); }
    result = cufftXtExecDescriptorC2C(plan_fft3, device_data_kernel, device_data_kernel, CUFFT_FORWARD);
    if (result != CUFFT_SUCCESS) { printf ("*XtExecC2C kernel failed\n"); exit (EXIT_FAILURE); }

    // multiply both ffts and scale output
    if (benchmark)
        printf("Matrix Multiply on multiple GPUs\n");

    for (int i = 0; i<nGPUs; ++i){
        cudaSetDevice(deviceNum[i]);
        cufftComplex *input_data_on_gpu, *kernel_data_on_gpu;
        input_data_on_gpu = (cufftComplex*) (device_data_input->descriptor->data[deviceNum[i]]);
        kernel_data_on_gpu = (cufftComplex*) (device_data_kernel->descriptor->data[deviceNum[i]]);
        // multiply, scale both arrays, keep product inplace on device_data_input cudaLibXtDesc
        int size_device = int(device_data_input->descriptor->size[deviceNum[i]] / sizeof(cufftComplex));

        if (benchmark) {
            printf("\n\nDevice: %d, elements: %d", deviceNum[i], size_device);
            printf("\ninput FFT deviceNum:%d\n", deviceNum[i]);
            /*cufftutils::printDeviceData(input_data_on_gpu, size_device);*/
            printf("\nkernel FFT deviceNum:%d\n", deviceNum[i]);
            /*cufftutils::printDeviceData(kernel_data_on_gpu, size_device);*/
        }

        // product is in-place for the second matrix passed (input)
        ComplexPointwiseMulAndScale<<<32, 256>>>((cufftComplex*) kernel_data_on_gpu, 
                (cufftComplex*) input_data_on_gpu, size_device, 1.0f / (float) N_padded);

        if (benchmark) {
            printf("\nProduct deviceNum:%d\n", deviceNum[i]);
            /*cufftutils::printDeviceData(input_data_on_gpu, size_device);*/
        }
    }

    // Synchronize GPUs
    for (int i = 0; i<nGPUs; ++i){
        cudaSetDevice(deviceNum[i]);
        cudaDeviceSynchronize();
    }

    // Perform inverse FFT on multiple GPUs
    if (benchmark)
        printf("Inverse 3d FFT on multiple GPUs\n");
    result = cufftXtExecDescriptorC2C(plan_fft3, device_data_input, device_data_input, CUFFT_INVERSE);
    if (result != CUFFT_SUCCESS) { printf ("*XtExecDesc inverse failed, code: %d\n",result); exit (EXIT_FAILURE); }

    if (benchmark) {
        cudaEventRecord(stop, 0);
        cudaDeviceSynchronize();
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        printf(" Full convolution cost %.2f ms\n", elapsed);
    }

     /*Copy the output data from multiple gpus to the 'host' result variable (automatically reorders the data from output to natural order)*/
    result = cufftXtMemcpy (plan_fft3, host_data_input, device_data_input, CUFFT_COPY_DEVICE_TO_HOST);
    if (result != CUFFT_SUCCESS) { printf ("*cufftXtMemcpy failed, code: %d\n",result); exit (EXIT_FAILURE); }

    if (benchmark) {
        /*printf("Print hostO final\n");*/
        /*cufftutils::printHostData(host_data_input, N_padded);*/
        printf("Place results in output\n");
    }


    cufftutils::trim_pad(trim_idxs, size, pad_size, column_order, hostO, host_data_input, benchmark);

    if (benchmark)
        printf("Cufft completed successfully\n");

    // Synchronize GPUs
    for (int i = 0; i<nGPUs; ++i){
        cudaSetDevice(deviceNum[i]);
        cudaDeviceSynchronize();
    }

    // Free malloc'ed variables
    free(worksize);
    // Free malloc'ed variables
    free(host_data_input);
    free(host_data_kernel);

    // Destroy FFT plan
    // must be destroyed to free enough memory for inverse
    result = cufftDestroy(plan_fft3);
    if (result != CUFFT_SUCCESS) { printf ("cufftDestroy failed: code %d\n",(int)result); exit (EXIT_FAILURE); }

    // Free cufftX malloc'ed variables
    result = cufftXtFree(device_data_input);
    if (result != CUFFT_SUCCESS) { printf ("*XtFree failed\n"); exit (EXIT_FAILURE); }
    result = cufftXtFree(device_data_kernel);
    if (result != CUFFT_SUCCESS) { printf ("*XtFree failed\n"); exit (EXIT_FAILURE); }

    return 0;
}

int fft3_test(float * data, unsigned int* size, unsigned int* length, float* outArray, bool column_order) {
     
    int nGPUs;
    cudaGetDeviceCount(&nGPUs);
    printf("No. of GPU on node %d\n", nGPUs);

    /*FIXME account for length*/
    int i, j, k, idx;
    int N = size[0] * size[1] * size[2];
    /*int max_thread = 1024;*/

    //Create complex variables on host
    cufftComplex *u = (cufftComplex *)malloc(sizeof(cufftComplex) * N);
    cufftComplex *u_fft = (cufftComplex *)malloc(sizeof(cufftComplex) * N);

    // Initialize the transform memory 
    for ( int i = 0; i < N; i++)
    {
        u[i].x = data[i];
        u[i].y = 0.0f;

        u_fft[i].x = 0.0f;
        u_fft[i].y = 0.0f;
    }

    // Set GPU's to use and list device properties
    int deviceNum[nGPUs];
    for(i = 0; i<nGPUs; ++i)
    {
        deviceNum[i] = i;
    }

    /*printf("Running Multi_GPU_FFT_check using %d GPUs on a %dx%dx%d grid.\n",nGPUs, size[0], size[1], size[2]);*/

    // Launch CUDA kernel to convert to complex
    /*cudaSetDevice(deviceNum[0]);*/
    /*initialize<<<N / max_thread + 1, max_thread>>>(N, data, u, u_fft);*/

    /*// Synchronize GPUs before moving forward*/
    /*for (i = 0; i<nGPUs; ++i){*/
        /*cudaSetDevice(deviceNum[i]);*/
        /*cudaDeviceSynchronize();*/
    /*}*/

    /*float elapsed = 0.0f;*/
    /*cudaEvent_t start, stop;*/
    /*cudaEventCreate(&start);*/
    /*cudaEventCreate(&stop);*/
    /*cudaEventRecord(start, 0);*/

    // Initialize CUFFT for multiple GPUs //
    // Initialize result variable used for error checking
    cufftResult result;

    // Create empty plan that will be used for the FFT
    cufftHandle plan;
    result = cufftCreate(&plan);
    if (result != CUFFT_SUCCESS) { printf ("*Create failed\n"); return 1; }

    // Tell cuFFT which GPUs to use
    result = cufftXtSetGPUs (plan, nGPUs, deviceNum);
    if (result != CUFFT_SUCCESS) { printf ("*XtSetGPUs failed: code %i\n", result); return 1; }

    // Create the plan for the FFT
    // Initializes the worksize variable
    size_t *worksize;                                   
    // Allocates memory for the worksize variable, which tells cufft how many GPUs it has to work with
    worksize =(size_t*)malloc(sizeof(size_t) * nGPUs);  
    
    // Create the plan for cufft, each element of worksize is the workspace for that GPU
    // multi-gpus must have a complex to complex transform
    result = cufftMakePlan3d(plan, size[0], size[1], size[2], CUFFT_C2C, worksize); 
    if (result != CUFFT_SUCCESS) { printf ("*MakePlan* failed: code %d \n",(int)result); exit (EXIT_FAILURE) ; }

    /*printf("The size of the worksize is %lu\n", worksize[0]);*/

    // Initialize transform array - to be split among GPU's and transformed in place using cufftX
    cudaLibXtDesc *device_data_input;
    // Allocate data on multiple gpus using the cufft routines
    result = cufftXtMalloc(plan, (cudaLibXtDesc **)&device_data_input, CUFFT_XT_FORMAT_INPLACE);
    if (result != CUFFT_SUCCESS) { printf ("*XtMalloc failed, code: %d\n", result); exit (EXIT_FAILURE) ; }

    // Copy the data from 'host' to device using cufftXt formatting
    result = cufftXtMemcpy(plan, device_data_input, u, CUFFT_COPY_HOST_TO_DEVICE);
    if (result != CUFFT_SUCCESS) { printf ("*XtMemcpy failed, code: %d\n",result); exit (EXIT_FAILURE); }

    // Perform FFT on multiple GPUs
    printf("Forward 3d FFT on multiple GPUs\n");
    result = cufftXtExecDescriptorC2C(plan, device_data_input, device_data_input, CUFFT_FORWARD);
    if (result != CUFFT_SUCCESS) { printf ("*XtExecC2C  failed\n"); exit (EXIT_FAILURE); }

    /*cudaEventRecord(stop, 0);*/
    /*cudaEventSynchronize(stop);*/
    /*cudaEventElapsedTime(&elapsed, start, stop);*/
    /*cudaEventDestroy(start);*/
    /*cudaEventDestroy(stop);*/
    /*printf("1 FFTs cost %.2f ms\n", elapsed);*/

    // Perform inverse FFT on multiple GPUs
    printf("Inverse 3d FFT on multiple GPUs\n");
    result = cufftXtExecDescriptorC2C(plan, device_data_input, device_data_input, CUFFT_INVERSE);
    if (result != CUFFT_SUCCESS) { printf ("*XtExecC2C  failed\n"); exit (EXIT_FAILURE); }

    // Copy the output data from multiple gpus to the 'host' result variable (automatically reorders the data from output to natural order)
    result = cufftXtMemcpy (plan, u_fft, device_data_input, CUFFT_COPY_DEVICE_TO_HOST);
    if (result != CUFFT_SUCCESS) { printf ("*XtMemcpy failed\n"); exit (EXIT_FAILURE); }

    // Scale output to match input (cuFFT does not automatically scale FFT output by 1/N)
    cudaSetDevice(deviceNum[0]);

    // cuFFT does not scale transform to 1 / N 
    for (int idx=0; idx < N; idx++) {
        u_fft[idx].x = u_fft[idx].x / ( (float)N );
        u_fft[idx].y = u_fft[idx].y / ( (float)N );
    }

    /*scaleResult<<<N / max_thread + 1, max_thread>>>(N, u_fft);*/

    /*for (i = 0; i<nGPUs; ++i){*/
        /*cudaSetDevice(deviceNum[i]);*/
        /*idx = i*NX_per_GPU*NY*NZ;                // sets the index value of the data to send to each gpu*/
        /*scaleResult<<<gridSize, blockSize>>>(NX_per_GPU, &u_fft[idx]);*/
    /*}*/

    /*// Synchronize GPUs*/
    /*for (i = 0; i<nGPUs; ++i){*/
        /*cudaSetDevice(deviceNum[i]);*/
        /*cudaDeviceSynchronize();*/
    /*}*/

    // Test results to make sure that u = u_fft
    float error = 0.0;
    for (i = 0; i<size[0]; ++i){
        for (j = 0; j<size[1]; ++j){
            for (k = 0; k<size[2]; ++k){
                idx = k + j*size[2] + size[2]*size[1]*i;
                // error += (float)u[idx].x - sin(x)*cos(y)*cos(z);
                error += abs((float)u[idx].x - (float)u_fft[idx].x);
                // printf("At idx = %d, the value of the error is %f\n",idx,(float)u[idx].x - (float)u_fft[idx].x);
                // printf("At idx = %d, the value of the error is %f\n",idx,error);

            }
        }
    }
    printf("The sum of the error is %4.4g\n",error);

    // Deallocate variables

    // Free malloc'ed variables
    free(worksize);
    // Free cuda malloc'ed variables
    cudaFree(u);
    cudaFree(u_fft);
    // Free cufftX malloc'ed variables
    result = cufftXtFree(device_data_input);
    if (result != CUFFT_SUCCESS) { printf ("*XtFree failed\n"); exit (EXIT_FAILURE); }
    // result = cufftXtFree(u_reorder);
    // if (result != CUFFT_SUCCESS) { printf ("*XtFree failed\n"); exit (EXIT_FAILURE); }
    // Destroy FFT plan
    result = cufftDestroy(plan);
    if (result != CUFFT_SUCCESS) { printf ("cufftDestroy failed: code %d\n",(int)result); exit (EXIT_FAILURE); }

    return 0;
}

int fft3(float * data, unsigned int* size, unsigned int* length, float* outArray, bool column_order) {
     
    int nGPUs;
    cudaGetDeviceCount(&nGPUs);

    /*FIXME account for length*/
    int i, j, k, idx;
    int N = size[0] * size[1] * size[2];
    /*int max_thread = 1024;*/

    //Create complex variables on host
    cufftComplex *u = (cufftComplex *)malloc(sizeof(cufftComplex) * N);
    cufftComplex *u_fft = (cufftComplex *)malloc(sizeof(cufftComplex) * N);

    // Initialize the transform memory 
    for ( int i = 0; i < N; i++)
    {
        u[i].x = data[i];
        u[i].y = 0.0f;

        u_fft[i].x = 0.0f;
        u_fft[i].y = 0.0f;
    }

    // Set GPU's to use and list device properties
    int deviceNum[nGPUs];
    for(i = 0; i<nGPUs; ++i)
    {
        deviceNum[i] = i;
    }

    /*printf("Running Multi_GPU_FFT_check using %d GPUs on a %dx%dx%d grid.\n",nGPUs, size[0], size[1], size[2]);*/

    // Launch CUDA kernel to convert to complex
    /*cudaSetDevice(deviceNum[0]);*/
    /*initialize<<<N / max_thread + 1, max_thread>>>(N, data, u, u_fft);*/

    /*// Synchronize GPUs before moving forward*/
    /*for (i = 0; i<nGPUs; ++i){*/
        /*cudaSetDevice(deviceNum[i]);*/
        /*cudaDeviceSynchronize();*/
    /*}*/

    // Initialize CUFFT for multiple GPUs //
    // Initialize result variable used for error checking
    cufftResult result;

    // Create empty plan that will be used for the FFT
    cufftHandle plan;
    result = cufftCreate(&plan);
    if (result != CUFFT_SUCCESS) { printf ("*Create failed\n"); return 1; }

    // Tell cuFFT which GPUs to use
    result = cufftXtSetGPUs (plan, nGPUs, deviceNum);
    if (result != CUFFT_SUCCESS) { printf ("*XtSetGPUs failed: code %i\n", result); return 1; }

    // Create the plan for the FFT
    // Initializes the worksize variable
    size_t *worksize;                                   
    // Allocates memory for the worksize variable, which tells cufft how many GPUs it has to work with
    worksize =(size_t*)malloc(sizeof(size_t) * nGPUs);  
    
    // Create the plan for cufft, each element of worksize is the workspace for that GPU
    // multi-gpus must have a complex to complex transform
    result = cufftMakePlan3d(plan, size[0], size[1], size[2], CUFFT_C2C, worksize); 
    if (result != CUFFT_SUCCESS) { printf ("*MakePlan* failed: code %d \n",(int)result); exit (EXIT_FAILURE) ; }

    /*printf("The size of the worksize is %lu\n", worksize[0]);*/

    // Initialize transform array - to be split among GPU's and transformed in place using cufftX
    cudaLibXtDesc *device_data_input;
    // Allocate data on multiple gpus using the cufft routines
    result = cufftXtMalloc(plan, (cudaLibXtDesc **)&device_data_input, CUFFT_XT_FORMAT_INPLACE);
    if (result != CUFFT_SUCCESS) { printf ("*XtMalloc failed, code: %d\n", result); exit (EXIT_FAILURE) ; }

    // Copy the data from 'host' to device using cufftXt formatting
    result = cufftXtMemcpy(plan, device_data_input, u, CUFFT_COPY_HOST_TO_DEVICE);
    if (result != CUFFT_SUCCESS) { printf ("*XtMemcpy failed, code: %d\n",result); exit (EXIT_FAILURE); }

    // Perform FFT on multiple GPUs
    printf("Forward 3d FFT on multiple GPUs\n");
    result = cufftXtExecDescriptorC2C(plan, device_data_input, device_data_input, CUFFT_FORWARD);
    if (result != CUFFT_SUCCESS) { printf ("*XtExecC2C  failed\n"); exit (EXIT_FAILURE); }

    // Perform inverse FFT on multiple GPUs
    printf("Inverse 3d FFT on multiple GPUs\n");
    result = cufftXtExecDescriptorC2C(plan, device_data_input, device_data_input, CUFFT_INVERSE);
    if (result != CUFFT_SUCCESS) { printf ("*XtExecC2C  failed\n"); exit (EXIT_FAILURE); }

    // Copy the output data from multiple gpus to the 'host' result variable (automatically reorders the data from output to natural order)
    result = cufftXtMemcpy (plan, u_fft, device_data_input, CUFFT_COPY_DEVICE_TO_HOST);
    if (result != CUFFT_SUCCESS) { printf ("*XtMemcpy failed\n"); exit (EXIT_FAILURE); }

    // Scale output to match input (cuFFT does not automatically scale FFT output by 1/N)
    cudaSetDevice(deviceNum[0]);

    // cuFFT does not scale transform to 1 / N 
    for (int idx=0; idx < N; idx++) {
        u_fft[idx].x = u_fft[idx].x / ( (float)N );
        u_fft[idx].y = u_fft[idx].y / ( (float)N );
    }

    /*scaleResult<<<N / max_thread + 1, max_thread>>>(N, u_fft);*/

    /*for (i = 0; i<nGPUs; ++i){*/
        /*cudaSetDevice(deviceNum[i]);*/
        /*idx = i*NX_per_GPU*NY*NZ;                // sets the index value of the data to send to each gpu*/
        /*scaleResult<<<gridSize, blockSize>>>(NX_per_GPU, &u_fft[idx]);*/
    /*}*/

    /*// Synchronize GPUs*/
    /*for (i = 0; i<nGPUs; ++i){*/
        /*cudaSetDevice(deviceNum[i]);*/
        /*cudaDeviceSynchronize();*/
    /*}*/

    // Test results to make sure that u = u_fft
    float error = 0.0;
    for (i = 0; i<size[0]; ++i){
        for (j = 0; j<size[1]; ++j){
            for (k = 0; k<size[2]; ++k){
                idx = k + j*size[2] + size[2]*size[1]*i;
                error += abs((float)u[idx].x - (float)u_fft[idx].x);
                // printf("At idx = %d, the value of the error is %f\n",idx,error);

            }
        }
    }
    printf("The sum of the error is %4.4g\n",error);

    // Deallocate variables

    // Free malloc'ed variables
    free(worksize);
    // Free cuda malloc'ed variables
    cudaFree(u);
    cudaFree(u_fft);
    // Free cufftX malloc'ed variables
    result = cufftXtFree(device_data_input);
    if (result != CUFFT_SUCCESS) { printf ("*XtFree failed\n"); exit (EXIT_FAILURE); }
    // result = cufftXtFree(u_reorder);
    // if (result != CUFFT_SUCCESS) { printf ("*XtFree failed\n"); exit (EXIT_FAILURE); }
    // Destroy FFT plan
    result = cufftDestroy(plan);
    if (result != CUFFT_SUCCESS) { printf ("cufftDestroy failed: code %d\n",(int)result); exit (EXIT_FAILURE); }

    return 0;
}


} // namespace cufftutils
