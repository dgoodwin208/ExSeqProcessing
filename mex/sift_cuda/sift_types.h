#ifndef __SIFT_TYPES_H__
#define __SIFT_TYPES_H__

#include <cuda_runtime.h>
#include <fstream>
#include <stdexcept>

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
    double* ivec; //stores the flattened descriptor vector
} Keypoint;

typedef struct _Keypoint_store {
    Keypoint * buf;
    int len;
} Keypoint_store;

// WARNING: Known compiler issue corrupts these values
// when passed to a kernel if new variables are
// added that cause misalignment
// placing variable types contiguously and
// in decreasing size order
typedef struct _SiftParams {
    double MagFactor;
    double IndexSize;
    double IndexSigma; 
    double Tessellation_levels;
    double SigmaScaled;
    double Tessel_thresh;
    double TwoPeak_Thresh;
    double Smooth_Var;
    double xyScale;
    double tScale;
    double MaxIndexVal;
    int fv_centers_len;
    int keypoint_num;
    int nFaces;
    int image_size0;
    int image_size1;
    int image_size2;
    int descriptor_len;
    bool Smooth_Flag;
    bool TwoPeak_Flag;
    bool skipDescriptor;

} SiftParams;


static double* sift_defaults(cudautils::SiftParams * sift_params,
            const unsigned int x_size,
            const unsigned int y_size,
            const unsigned int z_size,
            const unsigned int keypoint_num) {
        sift_params->image_size0 = x_size;
        sift_params->image_size1 = y_size;
        sift_params->image_size2 = z_size;
        sift_params->fv_centers_len = 240;
        sift_params->IndexSize = 2;
        sift_params->nFaces = 80;
        sift_params->IndexSigma = 5.0;
        sift_params->SigmaScaled = sift_params->IndexSigma * 0.5 * sift_params->IndexSize;
        sift_params->Smooth_Flag = true;
        sift_params->Smooth_Var = 20;
        sift_params->MaxIndexVal = 0.2;
        sift_params->Tessel_thresh = 3;
        sift_params->TwoPeak_Thresh = .9;
        sift_params->xyScale = 1;
        sift_params->tScale = 1;
        sift_params->TwoPeak_Flag = true;
        sift_params->skipDescriptor = false; // only record location of kypts
        sift_params->MagFactor = 3;
        sift_params->keypoint_num = keypoint_num; 
        sift_params->descriptor_len = sift_params->IndexSize *
            sift_params->IndexSize * sift_params->IndexSize * sift_params->nFaces;

        // set fv_centers
        std::vector<double> fv_centers(sift_params->fv_centers_len);

        //uint32_t fv_centers_len;
        unsigned int fv_centers_len;
        std::ifstream fin1("fv_centers.bin", std::ios::binary);
        if (fin1.is_open()) {
            fin1.read((char*)&fv_centers_len, sizeof(unsigned int));
            assert(fv_centers_len == sift_params->fv_centers_len);
            fin1.read((char*)fv_centers.data(), sift_params->fv_centers_len * sizeof(double));
            //for (int i=0; i < sift_params->fv_centers_len; i++) {
                //printf("%d %.1f \n", i, fv_centers[i]);
            //}
        } else { 
            throw std::invalid_argument( "Unable to open or find file: `fv_centers.bin` in current directory");
        }
        return &fv_centers[0];
}

}

#endif //__SIFT_TYPES_H__
