#ifndef __NEARESTINTERP_BRIDGE_H__
#define __NEARESTINTERP_BRIDGE_H__

#include <cstdint>
#include <memory>

#include "spdlog/spdlog.h"

namespace cudautils {

void nearestinterp_bridge(
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
        double* out_image);

}

#endif // __NEARESTINTERP_BRIDGE_H__

