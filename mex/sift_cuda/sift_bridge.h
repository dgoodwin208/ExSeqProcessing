#ifndef __SIFT_BRIDGE_H__
#define __SIFT_BRIDGE_H__

#include <cstdint>
#include <memory>

#include "spdlog/spdlog.h"
#include "sift_types.h"

namespace cudautils {

void sift_bridge(
        std::shared_ptr<spdlog::logger> logger,
        const unsigned int x_size,
        const unsigned int y_size,
        const unsigned int z_size,
        const unsigned int x_sub_size,
        const unsigned int y_sub_size,
        const unsigned int dx,
        const unsigned int dy,
        const unsigned int dw,
        const int num_gpus,
        const int num_streams,
        const double* in_image,
        const int8_t* in_map,
        const cudautils::SiftParams sift_params,
        const double* fv_centers,
        cudautils::Keypoint_store *keystore);

}

#endif // __SIFT_BRIDGE_H__

