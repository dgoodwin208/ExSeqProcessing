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
#include <stdexcept>
#include "error_helper.h"
#include "cuda_timer.h"

// includes, project
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime.h>
#include <sys/time.h>

/*#define DEBUG_OUTPUT*/

typedef thrust::host_vector<double, thrust::system::cuda::experimental::pinned_allocator<double>> pinnedDblHostVector;
typedef thrust::host_vector<unsigned int, thrust::system::cuda::experimental::pinned_allocator<int>> pinnedUIntHostVector;

namespace cufftutils {
    
    double cpuSecond() {
        struct timeval tp;
        gettimeofday(&tp, NULL);
        return((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
    }

// Pointwise Multiplication CUDA Kernel called by convolution3d
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
    printf("[%d]=%.3f %.3f\n", i, a[i].x, a[i].y);
}

void printDeviceData(cufftComplex *a, int size) {

  cufftComplex *h;

  h = (cufftComplex *) malloc(sizeof(cufftComplex) * size);

  cudaMemcpy(h, a, sizeof(cufftComplex) * size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < size; i++)
    printf("%f %f\n", h[i].x, h[i].y);

  free(h);
}

// 3D FFT on device
void complex_fft3_1GPU(cufftComplex *d_signal, int NX, int NY, int NZ,
        int direction) {

  cufftHandle plan;
  int n[] = {NX, NY, NZ};
  int NRANK = 3;
  int batch = 1;

  // if inembed and onembed are set to NULL, other advanced params are ignored
  cufftSafeCall(cufftPlanMany(&plan, NRANK, n,
              NULL, 1, NX*NY*NZ, // *inembed, istride, idist
              NULL, 1, NX*NY*NZ, // *onembed, ostride, odist
              CUFFT_C2C, batch));

  cufftSafeCall(cufftExecC2C(plan, d_signal, d_signal, direction) ); 
}

void cudaConvolution3D_1GPU(cufftComplex *d_signal1, unsigned int* size1, cufftComplex *d_signal2,
                unsigned int* size2, dim3 blockSize, dim3 gridSize, int benchmark) {

    int N = size1[0] * size1[1] * size1[2];
    if (benchmark)
        printf("cudaConvolution3D_1GPU");

    cufftutils::complex_fft3_1GPU(d_signal1, size1[0], size1[1], size1[2], CUFFT_FORWARD);

    cufftutils::complex_fft3_1GPU(d_signal2, size2[0], size2[1], size2[2], CUFFT_FORWARD);

    if (benchmark) {
        printf("\nManual input FFT\n");
        cufftutils::printDeviceData(d_signal1, N);
        printf("\nManual kernel FFT\n");
        cufftutils::printDeviceData(d_signal2, N);
    }

    pwProd<<<gridSize, blockSize>>>(d_signal1, N, d_signal2) ;
    if (benchmark) {
        printf("\nManual product \n");
        cufftutils::printDeviceData(d_signal1, N);
    }

    cufftutils::complex_fft3_1GPU(d_signal1, size1[0], size1[1], size1[2], CUFFT_INVERSE);
}

__global__ 
void initialize(int N, float* data, cufftComplex *f1, cufftComplex *f2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Initialize complex array from real input
        f1[idx].x = data[idx];
        f1[idx].y = 0.0; // complex component

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


/* 
   Compute pad length per given dimension lengths
   This can produce issues for cufft since Multi-GPUs have strict size reqs related to prime factors.
   For 2D and 3D transforms, the dimensions must factor into primes less than or equal to 127.
   In general, powers of two are fastest, but they can come with a prohibitive memory cost.
   Simply choosing the next power of two for each dimension could result in a data matrices 
   up to 8 times larger than original, this is unnacceptable. The next best performance 
   comes from numbers that are a multiples of the primes 2, 3, 5, and 7. Therefore multiples
   of these prime factors are chosen to save GPU memory for larger dimension sizes.

   */
long get_pad_idx(long m, long n) {
    long min = m + n - 1;
    long next = pow(2, ceil(log(min) / log(2)));
    if ((next == 512) && (min <= 140)) { return 140; }
    if ((next == 512) && (min <= 210)) { return 210; }
    if ((next == 1024) && (min <= 630)) { return 630; }
    if ((next == 2048) && (min <= 1260)) { return 1260; }
    if ((next == 4096) && (min <= 2100)) { return 2100; }
    return next;
}


long long convert_idx(long i, long j, long k, unsigned int *matrix_size, bool column_order) {
    if (column_order) {
        return i + j * matrix_size[0] + ((long long) k) * matrix_size[0] * matrix_size[1];
    } else {
        return k + j * matrix_size[2] + ((long long) i) * matrix_size[2] * matrix_size[1];
    }
}

__forceinline__ __device__ 
long long convert_idx_gpu(long i, long j, long k, unsigned int matrix_size0, 
        unsigned int matrix_size1, unsigned int matrix_size2, bool column_order) {
    if (column_order) {
        return i + j * matrix_size0 + ((long long) k) * matrix_size0 * matrix_size1;
    } else {
        return k + j * matrix_size2 + ((long long) i) * matrix_size2 * matrix_size1;
    }
}

// converts from column order to c-order when column_to_c != 0, otherwise reversed
void convert_matrix(float* matrix, float* buffer, unsigned int* size, bool column_order) {
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

__forceinline__ __device__
void ind2sub(const unsigned int x_stride, const unsigned int y_stride, const unsigned long long idx, unsigned int& x, unsigned int& y, unsigned int& z) {
    unsigned int i = idx;
    z = i / (x_stride * y_stride);
    i -= z * (x_stride * y_stride);

    y = i / x_stride;
    x = i - y * x_stride;
}

// column-major order since image is from matlab
__forceinline__ __device__
void ind2sub_col(const unsigned int x_stride, const unsigned int y_stride, const unsigned long long idx, unsigned int& x, unsigned int& y, unsigned int& z) {
    x = idx % x_stride;
    y = (idx - x)/x_stride % y_stride;
    z = ((idx - x)/x_stride - y)/y_stride;
}

__global__
void initialize_inputs_1GPU(float* src_img, float* src_flt, cufftComplex
        dst_img[], cufftComplex dst_flt[], long long size_device, long
        long start, unsigned int size0, unsigned int size1, unsigned int size2,
        unsigned int pad_size0, unsigned int pad_size1, unsigned int pad_size2,
        unsigned int filterdimA0, unsigned int filterdimA1, unsigned int
        filterdimA2, bool column_order, int benchmark) {
    
    long long thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= size_device) {return;}
    long long pad_image_idx = thread_idx + start;

    // identify corresponding index
    unsigned int i, j, k;
    ind2sub_col(pad_size0, pad_size1, pad_image_idx, i, j, k);

    // Place in matrix padded to 0
    long long idx;
    long long idx_filter;
    long long pad_idx;
    if ((i < pad_size0) && (j < pad_size1) && (k < pad_size2)) { 

        idx = convert_idx_gpu(i, j, k, size0, size1, size2, column_order);
        idx_filter = convert_idx_gpu(i, j, k, filterdimA0, filterdimA1, filterdimA2, column_order);
        // always place into c-order for cuda processing, revert in trim_pad()
        pad_idx = convert_idx_gpu(i, j, k, pad_size0, pad_size1, pad_size2, false); 

        if ((i < filterdimA0) && (j < filterdimA1) && (k < filterdimA2)) {
            dst_flt[pad_idx].x = src_flt[idx_filter];
        } else {
            dst_flt[pad_idx].x = 0.0f;
        }
        dst_flt[pad_idx].y = 0.0f; // y is complex component

        // keep in Matlab Column-order but switch order of dimensions in createPlan
        // to accomplish c-order FFT transforms
        if ((i < size0) && (j < size1) && (k < size2) ) {
            dst_img[pad_idx].x = src_img[idx];
        } else {
            dst_img[pad_idx].x = 0.0f;
        }
        dst_img[pad_idx].y = 0.0f; 
    }
}

void initialize_inputs(float* hostI, float* hostF, cufftComplex host_data_input[], 
        cufftComplex host_data_kernel[], unsigned int* size, unsigned int* pad_size, unsigned int* filterdimA,
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

void get_pad_trim(unsigned int* size, unsigned int* filterdimA, unsigned int* pad_size, int trim_idxs[3][2]) {
    // Compute pad lengths
    for (int i=0; i < 3; i++) 
        pad_size[i] = cufftutils::get_pad_idx(size[i], filterdimA[i]);

    for (int i=0; i < 3; i++) {
        trim_idxs[i][0] = ceil((filterdimA[i] - 1) / 2);
        trim_idxs[i][1] = size[i] + ceil((filterdimA[i] - 1) / 2);
    }
}

__global__
void trim_pad_1GPU(int trim_idxs00, int trim_idxs01, int trim_idxs10, 
        int trim_idxs11, int trim_idxs20, int trim_idxs21, unsigned int size0, unsigned int
        size1, unsigned int size2, unsigned int pad_size0, unsigned int
        pad_size1, unsigned int pad_size2, long long size_device, bool
        column_order, float* dst, cufftComplex* src, int benchmark) 
{

    long long thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= size_device) {return;}

    // identify corresponding index
    unsigned int i, j, k;
    ind2sub_col(pad_size0, pad_size1, thread_idx, i, j, k);

    long long idx;
    long long pad_idx;
    if ((trim_idxs00 <= i) && (i < trim_idxs01) && (trim_idxs10 <= j) && (j < trim_idxs11)
            && (trim_idxs20 <= k) && (k < trim_idxs21)) {

        idx = cufftutils::convert_idx_gpu(i - trim_idxs00,
                j - trim_idxs10, k - trim_idxs20, size0, size1, size2, column_order);
        // data always processed, stored in c-order, see initialize_inputs()
        pad_idx = cufftutils::convert_idx_gpu(i, j, k, pad_size0, pad_size1, pad_size2, false);

        dst[idx] = src[pad_idx].x;
        /*if (benchmark) {*/
            /*printf("i%d j%d k%d pad_idx:%lld idx:%lld %.3f\n", i, j, k, pad_idx,*/
                    /*idx, src[pad_idx].x);*/
        /*}*/
    }
}

void trim_pad(int trim_idxs[3][2], unsigned int* size, unsigned int* pad_size, bool column_order,
        float* hostO, cufftComplex* host_data_input, int benchmark) 
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
                    printf("i%d j%d k%d pad_idx:%lld idx:%lld %.3f\n", i, j, k, pad_idx,
                            idx, host_data_input[pad_idx].x);
            }
        }
    }
}

int conv_1GPU_handler(float* hostI, float* hostF, float* hostO, int algo, unsigned int* size, unsigned int* filterdimA, bool column_order, int benchmark) {

    if (((size[0] + filterdimA[0] - 1) < 32) || ( (size[1] + filterdimA[1] - 1) < 32 )) {
        throw std::invalid_argument("FFT can not compute with data less than 32 for the x and y dimension");
    }

    long long N = ((long long) size[0]) * size[1] * size[2];
    if (benchmark)
        printf("Using %d GPUs on a %dx%dx%d grid, N:%lld\n",1, size[0], size[1], size[2], N);

    unsigned int pad_size[3];
    int trim_idxs[3][2];
    cufftutils::get_pad_trim(size, filterdimA, pad_size, trim_idxs);

    long long N_padded = pad_size[0] * pad_size[1] * pad_size[2];
    long long size_of_data = N_padded * sizeof(cufftComplex);

    if (benchmark)
        printf("Padded to a %dx%dx%d grid, N:%lld\n",pad_size[0], pad_size[1], pad_size[2], N_padded);

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
        /*cufftutils::printDeviceData(host_data_input, N_padded);*/
        printf("\n1GPU host_data_kernel elements:%d\n", N_padded);
        /*cufftutils::printDeviceData(host_data_kernel, N_padded);*/
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
    cufftutils::cudaConvolution3D_1GPU(device_data_input, pad_size, device_data_kernel, pad_size, 
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

/* 
   Main 3D convolution function for multiple GPUs
   Entry point for conv_cuda.cpp mexfile
*/
int conv_handler(float* hostI, float* hostF, float* hostO, int algo, unsigned int* size, unsigned int* filterdimA, bool column_order, int benchmark) {

    if (((size[0] + filterdimA[0] - 1) < 32) || ( (size[1] + filterdimA[1] - 1) < 32 )) {
        throw std::invalid_argument("FFT can not compute with data less than 32 for the x and y dimension");
    }

    unsigned int pad_size[3];
    int trim_idxs[3][2];
    cufftutils::get_pad_trim(size, filterdimA, pad_size, trim_idxs);

    long long N = ((long long) size[0]) * size[1] * size[2];
    long long N_padded = pad_size[0] * pad_size[1] * pad_size[2];
    long long N_kernel = filterdimA[0] * filterdimA[1] * filterdimA[2];
    long long size_of_data = N_padded * sizeof(cufftComplex);
    long long float_size = N* sizeof(float);
    long long kernel_float_size = N_kernel* sizeof(float);
    double time_start;
    if (benchmark)
        time_start = cufftutils::cpuSecond(); //gpu_timer fails across multi-gpu context calls below

    int nGPUs;
    cudaGetDeviceCount(&nGPUs);

    // Set GPU's to use 
    int deviceNum[nGPUs];
    for(int i = 0; i<nGPUs; ++i)
    {
        deviceNum[i] = i;
    }

    // declare empty plan that will be used for FFT / IFFT
    cufftHandle plan_fft3;
    cufftSafeCall(cufftCreate(&plan_fft3));

    // Tell cuFFT which GPUs to use
    cufftSafeCall(cufftXtSetGPUs (plan_fft3, nGPUs, deviceNum));

    // Allocates memory for the worksize variable, which tells cufft how many GPUs it has to work with
    size_t *worksize;                                   
    worksize =(size_t*)malloc(sizeof(size_t) * nGPUs);  
    
    if (benchmark) {
        printf("Using %d GPUs on a %dx%dx%d padded grid, N_padded:%d\n",nGPUs, pad_size[0], pad_size[1], pad_size[2], N_padded);
    }

    // Create the plan for cufft, each element of worksize is the workspace for that GPU
    // multi-gpus must have a complex to complex transform
    cufftSafeCall(cufftMakePlan3d(plan_fft3, pad_size[0], pad_size[1], pad_size[2], CUFFT_C2C, worksize)); 

    float* devI, *devF;
    cudaMalloc(&devI, float_size); cudaCheckPtr(devI);
    cudaMalloc(&devF, kernel_float_size); cudaCheckPtr(devF);
    cudaSafeCall(cudaMemcpy(devI, hostI, float_size, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(devF, hostF, kernel_float_size, cudaMemcpyHostToDevice));

    //Create complex variables on host, used temp for transfer to hostO
    cufftComplex *host_data_output = (cufftComplex *)malloc(size_of_data);
    cudaCheckPtr(host_data_output);

    // Allocate data on multiple gpus using the cufft routines
    // Initialize transform array - to be split among GPU's and transformed in place using cufftX
    cudaLibXtDesc *device_data_input, *device_data_kernel;
    cufftSafeCall(cufftXtMalloc(plan_fft3, &device_data_input, CUFFT_XT_FORMAT_INPLACE));
    cufftSafeCall(cufftXtMalloc(plan_fft3, &device_data_kernel, CUFFT_XT_FORMAT_INPLACE));

    cufftComplex *input_data_on_gpu, *kernel_data_on_gpu;
    cudaMalloc(&input_data_on_gpu, size_of_data); cudaCheckPtr(input_data_on_gpu);
    cudaMalloc(&kernel_data_on_gpu, size_of_data); cudaCheckPtr(kernel_data_on_gpu);

    long long start = 0;
    long long blockSize = 256;
    long long gridSize;
    gridSize = (N_padded + blockSize - 1) / blockSize; // round up

    if (benchmark) {
        double time_elapsed = cufftutils::cpuSecond() - time_start;
        printf("Malloc input and output: %.4f s\n", time_elapsed);
        time_start = cufftutils::cpuSecond(); //gpu_timer fails across multi-gpu context calls below
    }

    cufftutils::initialize_inputs_1GPU<<<gridSize, blockSize>>>(devI, devF, input_data_on_gpu, 
            kernel_data_on_gpu, N_padded, start, size[0], size[1], size[2], pad_size[0], pad_size[1], 
            pad_size[2], filterdimA[0], filterdimA[1], filterdimA[2], column_order, benchmark);
    cudaCheckError();

    cudaSafeCall(cudaDeviceSynchronize());

    // place data manually into the cudaXtDesc
    long long start_kernel = 0;
    for (int i = 0; i<nGPUs; ++i){
        long long size_device = long(device_data_input->descriptor->size[deviceNum[i]] / sizeof(cufftComplex));
        long long size_device_kernel = long(device_data_kernel->descriptor->size[deviceNum[i]] / sizeof(cufftComplex));
        cudaSafeCall(cudaMemcpy(device_data_input->descriptor->data[deviceNum[i]],
                    input_data_on_gpu + start,
                    device_data_input->descriptor->size[deviceNum[i]],
                    cudaMemcpyDeviceToDevice));
        cudaSafeCall(cudaMemcpy(device_data_kernel->descriptor->data[deviceNum[i]],
                    kernel_data_on_gpu + start_kernel,
                    device_data_kernel->descriptor->size[deviceNum[i]],
                    cudaMemcpyDeviceToDevice));
        if (benchmark) {
            printf("start[%d]=%d, length:%d\n", i, start, size_device);
            /*cufftutils::printHostData(host_data_input + start,
              device_data_input->descriptor->size[deviceNum[i]] / sizeof(cufftComplex));*/
        }
        start += size_device;
        start_kernel = size_device_kernel;
    }

    if (benchmark) {
        double time_elapsed = cufftutils::cpuSecond() - time_start;
        printf("Input and output successfully initialized: %.1f s\n", time_elapsed);
        time_start = cufftutils::cpuSecond(); //gpu_timer fails across multi-gpu context calls below
    }
    // Perform FFT on multiple GPUs
    cufftSafeCall(cufftXtExecDescriptorC2C(plan_fft3, device_data_input, device_data_input, CUFFT_FORWARD));
    cufftSafeCall(cufftXtExecDescriptorC2C(plan_fft3, device_data_kernel, device_data_kernel, CUFFT_FORWARD));

    if (benchmark) {
        // Synchronize GPUs
        for (int i = 0; i<nGPUs; ++i){
            cudaSetDevice(deviceNum[i]);
            cudaDeviceSynchronize();
        }

        double time_elapsed = cufftutils::cpuSecond() - time_start;
        printf("Forward 3d FFT kernel and image on multiple GPUs: %.4f s\n", time_elapsed);
        time_start = cufftutils::cpuSecond(); //gpu_timer fails across multi-gpu context calls below
    }

    // multiply both ffts and scale output
    cudaStream_t streams[nGPUs];
    for (int i = 0; i<nGPUs; ++i){
        cudaSafeCall(cudaSetDevice(deviceNum[i]));
        cudaStream_t current_stream = streams[i];
        cudaSafeCall(cudaStreamCreateWithFlags(&current_stream,
                    cudaStreamNonBlocking));

        cufftComplex *input_data_on_gpu, *kernel_data_on_gpu;
        input_data_on_gpu = (cufftComplex*) (device_data_input->descriptor->data[deviceNum[i]]);
        kernel_data_on_gpu = (cufftComplex*) (device_data_kernel->descriptor->data[deviceNum[i]]);
        // multiply, scale both arrays, keep product inplace on device_data_input cudaLibXtDesc
        int size_device = int(device_data_input->descriptor->size[deviceNum[i]] / sizeof(cufftComplex));
        gridSize = (size_device + blockSize - 1) / blockSize; // round up

#ifdef DEBUG_OUTPUT
        printf("Device: %d, elements: %d\n", deviceNum[i], size_device);
        printf("\nInput FFT deviceNum:%d\n", deviceNum[i]);
        cufftutils::printDeviceData(input_data_on_gpu, size_device);
        printf("\nKernel FFT deviceNum:%d\n", deviceNum[i]);
        cufftutils::printDeviceData(kernel_data_on_gpu, size_device);
#endif

        // product is in-place for the second matrix passed (input)
        ComplexPointwiseMulAndScale<<<gridSize, blockSize, 0, current_stream>>>((cufftComplex*) kernel_data_on_gpu,
                    (cufftComplex*) input_data_on_gpu, size_device, 1.0f /
                    (float) N_padded);
        cudaCheckError();

#ifdef DEBUG_OUTPUT
        printf("\nProduct deviceNum:%d\n", deviceNum[i]);
        cufftutils::printDeviceData(input_data_on_gpu, size_device);
#endif

    }

    // Synchronize GPUs
    for (int i = 0; i<nGPUs; ++i){
        cudaSafeCall(cudaSetDevice(deviceNum[i]));
        cudaSafeCall(cudaDeviceSynchronize());
    }

    if (benchmark) {
        double time_elapsed = cufftutils::cpuSecond() - time_start;
        printf("Complex Matrix Multiply 3d FFT kernel and image on multiple GPUs: %.4f s\n", 
               time_elapsed);
        time_start = cufftutils::cpuSecond(); //gpu_timer fails across multi-gpu context calls below
    }

    // Perform inverse FFT on multiple GPUs
    cufftSafeCall(cufftXtExecDescriptorC2C(plan_fft3, device_data_input, device_data_input, CUFFT_INVERSE));

    if (benchmark) {
        double time_elapsed = cufftutils::cpuSecond() - time_start;
        printf("Inverse 3d FFT kernel and image on multiple GPUs: %.4f s\n", time_elapsed);
        time_start = cufftutils::cpuSecond();
    }

    // memcpy data manually from cudaXtDesc
    start = 0;
    for (int i = 0; i<nGPUs; ++i){
        long long size_device = long(device_data_input->descriptor->size[deviceNum[i]] / sizeof(cufftComplex));
        cudaSafeCall(cudaMemcpy( input_data_on_gpu + start,
                    device_data_input->descriptor->data[deviceNum[i]],
                    device_data_input->descriptor->size[deviceNum[i]], //size in bytes
                    cudaMemcpyDeviceToDevice));
        if (benchmark) {
            printf("start[%d]=%d, length:%d\n", i, start, size_device);
            /*cufftutils::printHostData(host_data_input + start,
              device_data_input->descriptor->size[deviceNum[i]] / sizeof(cufftComplex));*/
        }
        start += size_device;
    }

    if (benchmark) {
        double time_elapsed = cufftutils::cpuSecond() - time_start;
        printf("Memcpy image from XtDesc: %.4f s\n", time_elapsed);
        time_start = cufftutils::cpuSecond();
    }

    gridSize = (N_padded + blockSize - 1) / blockSize; // round up
    cufftutils::trim_pad_1GPU<<<gridSize, blockSize>>>(trim_idxs[0][0],
            trim_idxs[0][1], trim_idxs[1][0], trim_idxs[1][1], trim_idxs[2][0],
            trim_idxs[2][1], size[0], size[1], size[2], pad_size[0],
            pad_size[1], pad_size[2], N_padded, column_order, devI, input_data_on_gpu,
            benchmark);
    cudaDeviceSynchronize();

    if (benchmark) {
        double time_elapsed = cufftutils::cpuSecond() - time_start;
        printf("`trim_pad_1GPU`: %.4f s\n", time_elapsed);
    }

    cudaSafeCall(cudaMemcpy(hostO, devI, float_size, cudaMemcpyDeviceToHost));

    if (benchmark) {
        printf("Cufft completed successfully\n");
    }

#ifdef DEBUG_OUTPUT
        printf("Print hostO final\n");
        cufftutils::printHostData(hostO, N);
#endif

    // Free malloc'ed variables
    free(worksize);
    // Free malloc'ed variables
    free(host_data_output);
    cudaSafeCall(cudaFree(devI));
    cudaSafeCall(cudaFree(devF));
    cudaSafeCall(cudaFree(input_data_on_gpu));
    cudaSafeCall(cudaFree(kernel_data_on_gpu));

    // Destroy FFT plan
    cufftSafeCall(cufftDestroy(plan_fft3));

    // Free cufftX malloc'ed variables
    cufftSafeCall(cufftXtFree(device_data_input));
    cufftSafeCall(cufftXtFree(device_data_kernel));

    return 0;
}

} // namespace cufftutils
