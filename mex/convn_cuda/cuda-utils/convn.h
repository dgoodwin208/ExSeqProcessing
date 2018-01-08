#ifndef __CONVN_H__
#define __CONVN_H__

#include <vector>

namespace cudautils {

float* convn(float *image, const int channels, const int height, const int width, float *kernel, const int kernel_height, const int kernel_width);

}

#endif
