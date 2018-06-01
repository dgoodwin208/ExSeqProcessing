#ifndef __SIFT_TYPES_H__
#define __SIFT_TYPES_H__

// error handling code, derived from funcs in old cutil lib

#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define cudaCheckError() __cudaCheckError(__FILE__, __LINE__)

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err )
    {  
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
    return;
}

inline void __cudaCheckError( const char *file, const int line)
{
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString( err ) );
        exit (-1);
    }

    // Warning this can sig. lower performance of code
    // make sure this section is not executed in production binaries
#ifdef DEBUG_OUTPUT
    //err = cudaDeviceSynchronize();
    //if (cudaSuccess != err)
    //{
        //fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                //file, line, cudaGetErrorString( err ) );
        //exit( -1 );
    //}
#endif
    
    return;
}

namespace cudautils {

typedef struct _Keypoint {
    int x;
    int y;
    int z;
    double xyScale;
    double tScale;
    uint8_t* ivec; //stores the flattened descriptor vector
} Keypoint;

typedef struct _Keypoint_store {
    Keypoint * buf;
    int len;
} Keypoint_store;

typedef struct _SiftParams {
    double MagFactor;
    int IndexSize;
    int nFaces;
    int Tessellation_levels;
    int Smooth_Flag;
    double SigmaScaled;
    double Tessel_thresh;
    double Smooth_Var;
    int TwoPeak_Flag;
    double xyScale;
    double tScale;
    double MaxIndexVal;
    //FIXME must be in row order
    double* fv_centers;
    int fv_centers_len;
    int image_size0;
    int image_size1;
    int image_size2;
    int keypoint_num;
    int descriptor_len;
} SiftParams;

}

#endif //__SIFT_TYPES_H__
