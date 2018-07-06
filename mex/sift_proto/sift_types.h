#ifndef __SIFT_TYPES_H__
#define __SIFT_TYPES_H__

#include <cuda_runtime.h>

#define X_NAME "x"
#define Y_NAME "y"
#define Z_NAME "z"
#define TSCALE_NAME "tScale"
#define XYSCALE_NAME "xyScale"
#define IVEC_NAME "ivec"
#define IM_NDIMS 3 // number of dimensions in image

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
    double IndexSize;
    double IndexSigma; 
    double nFaces;
    double Tessellation_levels;
    bool Smooth_Flag;
    double SigmaScaled;
    double Tessel_thresh;
    double Smooth_Var;
    bool TwoPeak_Flag;
    double xyScale;
    double tScale;
    double MaxIndexVal;
    double* fv_centers;
    double fv_centers_len;
    int image_size0;
    int image_size1;
    int image_size2;
    double keypoint_num;
    int descriptor_len;

} SiftParams;

}

#endif //__SIFT_TYPES_H__
