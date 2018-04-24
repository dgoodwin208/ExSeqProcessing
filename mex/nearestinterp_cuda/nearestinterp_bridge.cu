#include <memory>

#include "nearestinterp_bridge.h"
#include "nearestinterp.h"
#include "cuda_task_executor.h"

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
        double* out_image) {

    std::shared_ptr<cudautils::NearestInterp> ni =
        std::make_shared<cudautils::NearestInterp>(x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams);

    cudautils::CudaTaskExecutor executor(num_gpus, num_streams, ni);

    logger->info("setImage start");
    ni->setImage(in_image);
    logger->info("setImage end");
    ni->setMapToBeInterpolated(in_map);
    logger->info("setMap end");

    logger->info("calc start");
    executor.run();
    logger->info("calc end");

    logger->info("getImage start");
    ni->getImage(out_image);
    logger->info("getImage end");

}

}

