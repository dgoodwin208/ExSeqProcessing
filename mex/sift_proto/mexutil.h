#ifndef __MEXUTIL_H__
#define __MEXUTIL_H__

#include "sift_types.h"

namespace cudautils {

static void init_Keypoint_store(cudautils::Keypoint_store * kp, cudautils::SiftParams sift_params) {
    kp->len = sift_params.keypoint_num;
    kp->buf = (cudautils::Keypoint*) malloc(kp->len * sizeof(cudautils::Keypoint));
}

}

#endif 
