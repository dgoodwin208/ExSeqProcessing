#ifndef __ERROR_HELPER_H__
#define __ERROR_HELPER_H__

#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime.h>

// error handling code, derived from funcs in old cutil lib
#define cufftSafeCall(err) __cufftSafeCall(err, __FILE__, __LINE__)
#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define cudaCheckError() __cudaCheckError(__FILE__, __LINE__)
#define cudaCheckPtr(ptr) __cudaCheckPtr(ptr, __FILE__, __LINE__)
#define cudaCheckPtrDevice(ptr) __cudaCheckPtrDevice(ptr, __FILE__, __LINE__)

static const char *_cudaGetErrorEnum(cufftResult error) {
    switch (error) {
        case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}

inline void __cufftSafeCall(cufftResult err, const char *file, const int line)
{
    if (CUFFT_SUCCESS != err )
    {  
        fprintf( stderr, "cufftSafeCall() failed at %s:%i\n\tError:%d %s.\tExiting...\n",
                file, line, err, _cudaGetErrorEnum( err ) );
        cudaDeviceReset(); exit(1);
    }
}

inline 
void __cudaCheckPtr(void* ptr, const char* file, const int line)
{
    if (ptr == NULL) {
        printf("Error: Null Ptr. Exiting in %s at line %s", file, line );
        exit(1);
    }
}

__device__
inline 
void __cudaCheckPtrDevice(void* ptr, const char* file, const int line)
{
    if (ptr == NULL) {
        printf("Error: exiting all threads in %s at line %s", file, line );
        asm("trap;");
    }
}

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err )
    {  
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString( err ) );
        cudaDeviceReset(); exit(1);
    }
    return;
}

inline void __cudaCheckError( const char *file, const int line)
{
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString( err ) );
        cudaDeviceReset(); exit(1);
    }

    // check for asynchronous errors during execution of kernel
    // Warning this can sig. lower performance of code
    // make sure this section is not executed in production binaries
#ifdef DEBUG_OUTPUT
    /*err = cudaDeviceSynchronize();*/
    //if (cudaSuccess != err)
    //{
        //fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                //file, line, cudaGetErrorString( err ) );
        //cudaDeviceReset(); exit(1);
    //}
#endif
    
    return;
}

#endif
