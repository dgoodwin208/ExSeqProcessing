// Multiple GPU version of cuFFT_check that uses multiple GPU's
// This program creates a real-valued 3D function sin(x)*cos(y)*cos(z) and then 
// takes the forward and inverse Fourier Transform, with the necessary scaling included. 
// The output of this process should match the input function

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>

// includes, project
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cuComplex.h>
//CUFFT Header file
#include <cufftXt.h>

/*namespace cufftutils {*/

int divUp(int a, int b) { return (a + b - 1) / b; }

__device__
int idxClip(int idx, int idxMax){
    return idx > (idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

__device__
int flatten(int col, int row, int stack, int width, int height, int depth){
    return idxClip(stack, depth) + idxClip(row, height)*depth + idxClip(col, width)*depth*height;
    // Note: using column-major indexing format
}

/*__global__ */
/*void initialize(int NX_per_GPU, int gpuNum, float* data, cufftComplex *f1, cufftComplex *f2)*/
/*{*/
    /*const int i = blockIdx.x * blockDim.x + threadIdx.x;*/
    /*const int j = blockIdx.y * blockDim.y + threadIdx.y;*/
    /*const int k = blockIdx.z * blockDim.z + threadIdx.z;*/
    /*if ((i >= NX_per_GPU) || (j >= NY) || (k >= NZ)) return;*/
    /*const int idx = flatten(i, j, k, NX, NY, NZ);*/

    /*// Create physical vectors in temporary memory*/
    /*float x = i * (float)L / NX + (float)gpuNum*NX_per_GPU*L / NX;*/
    /*float y = j * (float)L / NY;*/
    /*float z = k * (float)L / NZ;*/

    /*// Initialize starting array*/
    /*[>f1[idx].x = sin(x)*cos(y)*cos(z);<]*/
    /*f1[idx].x = data[x][y][z];*/
    /*f1[idx].y = 0.0;*/

    /*f2[idx].x = 0.0;*/
    /*f2[idx].y = 0.0;*/

    /*return;*/
/*}*/

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

/*int conv_handler(float* hostI, float* hostF, float* hostO, int algo, int* dimA, int* filterdimA, int benchmark) {*/

int fft3(float * data, int* size, int* length) {
     
    /*FIXME account for length*/
    int i, j, k, idx;
    int N = size[0] * size[1] * size[2];
    int max_thread = 1024;
    // float complex test;

    // Set GPU's to use and list device properties
    int nGPUs = 2, deviceNum[nGPUs];
    for(i = 0; i<nGPUs; ++i)
    {
        deviceNum[i] = i;

        cudaSetDevice(deviceNum[i]);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceNum[i]);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }

    printf("Running Multi_GPU_FFT_check using %d GPUs on a %dx%dx%d grid.\n",nGPUs, size[0], size[1], size[2]);

    // Initialize input data
    // Split data according to number of GPUs
    /*NX_per_GPU = size[0]/nGPUs;              // This is not a good solution long-term; needs more work for arbitrary grid sizes/nGPUs*/

    // Declare variables
    cufftComplex *u;
    cufftComplex *u_fft;

    // Allocate memory for arrays
    cudaMallocManaged(&u, sizeof(cufftComplex)*N );
    cudaMallocManaged(&u_fft, sizeof(cufftComplex)*N );

    // Launch CUDA kernel to convert to complex
    cudaSetDevice(deviceNum[0]);
    initialize<<<N / max_thread + 1, max_thread>>>(N, data, u, u_fft);
    /*for (i = 0; i<nGPUs; ++i){*/
        /*cudaSetDevice(deviceNum[i]);*/
        /*initialize<<<N / max_thread + 1, max_thread>>>(N, data, &u[idx], &u_fft[idx]);*/
    /*}*/

    /*// Launch CUDA kernel to initialize velocity field*/
    /*const dim3 blockSize(TX, TY, TZ);*/
    /*const dim3 gridSize(divUp(NX_per_GPU, TX), divUp(NY, TY), divUp(NZ, TZ));*/
    /*for (i = 0; i<nGPUs; ++i){*/
        /*cudaSetDevice(deviceNum[i]);*/
        /*int idx = i*NX_per_GPU*NY*NZ;                // sets the index value of the data to send to each gpu*/
        /*initialize<<<gridSize, blockSize>>>(NX_per_GPU, deviceNum[i], &u[idx], &u_fft[idx]);*/
    /*}*/

    // Synchronize GPUs before moving forward
    for (i = 0; i<nGPUs; ++i){
        cudaSetDevice(deviceNum[i]);
        cudaDeviceSynchronize();
    }

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
    size_t *worksize;                                   // Initializes the worksize variable
    worksize =(size_t*)malloc(sizeof(size_t) * nGPUs);  // Allocates memory for the worksize variable, which tells cufft how many GPUs it has to work with
    
    // Create the plan for cufft, each element of worksize is the workspace for that GPU
    result = cufftMakePlan3d(plan, size[0], size[1], size[2], CUFFT_Z2Z, worksize); // multi-gpus must have a complex to complex transform
    if (result != CUFFT_SUCCESS) { printf ("*MakePlan* failed: code %d \n",(int)result); exit (EXIT_FAILURE) ; }

    printf("The size of the worksize is %lu\n", worksize[0]);

    // Initialize transform array - to be split among GPU's and transformed in place using cufftX
    cudaLibXtDesc *u_prime;
    // Allocate data on multiple gpus using the cufft routines
    result = cufftXtMalloc(plan, (cudaLibXtDesc **)&u_prime, CUFFT_XT_FORMAT_INPLACE);
    if (result != CUFFT_SUCCESS) { printf ("*XtMalloc failed, code: %d\n", result); exit (EXIT_FAILURE) ; }

    // Copy the data from 'host' to device using cufftXt formatting
    result = cufftXtMemcpy(plan, u_prime, u, CUFFT_COPY_HOST_TO_DEVICE);
    if (result != CUFFT_SUCCESS) { printf ("*XtMemcpy failed, code: %d\n",result); exit (EXIT_FAILURE); }

    // Perform FFT on multiple GPUs
    printf("Forward 3d FFT on multiple GPUs\n");
    result = cufftXtExecDescriptorZ2Z(plan, u_prime, u_prime, CUFFT_FORWARD);
    if (result != CUFFT_SUCCESS) { printf ("*XtExecZ2Z  failed\n"); exit (EXIT_FAILURE); }

////////// Apparently re-ordering the data prior to the IFFT is not necessary (gives incorrect results)////////////////////
    // cudaLibXtDesc *u_reorder;
    // result = cufftXtMalloc(plan, (cudaLibXtDesc **)&u_reorder, CUFFT_XT_FORMAT_INPLACE);
    // if (result != CUFFT_SUCCESS) { printf ("*XtMalloc failed\n"); exit (EXIT_FAILURE) ; }
    // // Re-order data on multiple GPUs to natural order
    // printf("Reordering the data on the GPUs\n");
    // result = cufftXtMemcpy (plan, u_reorder, u_prime, CUFFT_COPY_DEVICE_TO_DEVICE);
    // if (result != CUFFT_SUCCESS) { printf ("*XtMemcpy failed\n"); exit (EXIT_FAILURE); }
/////////////////////////////////////////////////////////////////////////////////////////////

    // Perform inverse FFT on multiple GPUs
    printf("Inverse 3d FFT on multiple GPUs\n");
    result = cufftXtExecDescriptorZ2Z(plan, u_prime, u_prime, CUFFT_INVERSE);
    if (result != CUFFT_SUCCESS) { printf ("*XtExecZ2Z  failed\n"); exit (EXIT_FAILURE); }

    // Copy the output data from multiple gpus to the 'host' result variable (automatically reorders the data from output to natural order)
    result = cufftXtMemcpy (plan, u_fft, u_prime, CUFFT_COPY_DEVICE_TO_HOST);
    if (result != CUFFT_SUCCESS) { printf ("*XtMemcpy failed\n"); exit (EXIT_FAILURE); }

    // Scale output to match input (cuFFT does not automatically scale FFT output by 1/N)
    cudaSetDevice(deviceNum[0]);
    scaleResult<<<N / max_thread + 1, max_thread>>>(N, u_fft);
    /*for (i = 0; i<nGPUs; ++i){*/
        /*cudaSetDevice(deviceNum[i]);*/
        /*idx = i*NX_per_GPU*NY*NZ;                // sets the index value of the data to send to each gpu*/
        /*scaleResult<<<gridSize, blockSize>>>(NX_per_GPU, &u_fft[idx]);*/
    /*}*/

    // Synchronize GPUs
    for (i = 0; i<nGPUs; ++i){
        cudaSetDevice(deviceNum[i]);
        cudaDeviceSynchronize();
    }

    // Test results to make sure that u = u_fft
    float error = 0.0;
    for (i = 0; i<size[0]; ++i){
        for (j = 0; j<size[1]; ++j){
            for (k = 0; k<size[2]; ++k){
                idx = k + j*size[2] + size[2]*size[1]*i;
                // error += (float)u[idx].x - sin(x)*cos(y)*cos(z);
                error += (float)u[idx].x - (float)u_fft[idx].x;
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
    result = cufftXtFree(u_prime);
    if (result != CUFFT_SUCCESS) { printf ("*XtFree failed\n"); exit (EXIT_FAILURE); }
    // result = cufftXtFree(u_reorder);
    // if (result != CUFFT_SUCCESS) { printf ("*XtFree failed\n"); exit (EXIT_FAILURE); }
    // Destroy FFT plan
    result = cufftDestroy(plan);
    if (result != CUFFT_SUCCESS) { printf ("cufftDestroy failed: code %d\n",(int)result); exit (EXIT_FAILURE); }

    return 0;

}

int main (void)
{
    int size[3] = {2048, 2048, 141};
    int N = size[0] * size[1] * size[2];
    
    printf("Initializing rand array\n");
    float* data = new float[N]; 

    for (int i=0; i < N; i++)
        data[i] = rand() % 100;

    printf("Rand array created\n");

    int result = fft3(data, size, size);

    return 0;
}

/*} // namespace cufftutils*/
