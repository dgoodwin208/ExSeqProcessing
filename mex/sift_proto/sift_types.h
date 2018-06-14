#ifndef __SIFT_TYPES_H__
#define __SIFT_TYPES_H__

#include <cuda_runtime.h>

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
