/*=================================================================
 * nearestinterp_cuda.cpp - interpolate volume image data from the nearest intensities
 *
 *  nearestinterp_cuda(vol_image, map)
 *
 *  Input:
 *    vol_image:  volume image data
 *    map:        mask map data (1: mask, 0: hole)
 *
 *  Output:
 *    interpolated_image:  interpolated image data
 *
 *=================================================================*/
 

#include <fstream>
#include <vector>
#include <algorithm>

#include "gpudevice.h"
#include "nearestinterp_bridge.h"

#include "spdlog/spdlog.h"

#include "mex.h"
#include "matrix.h"

void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{ 
    std::shared_ptr<spdlog::logger> logger;
    try {
        spdlog::set_async_mode(4096, spdlog::async_overflow_policy::block_retry, nullptr, std::chrono::seconds(2));
        spdlog::set_level(spdlog::level::trace);
        logger = spdlog::get("mex_logger");
        if (logger == nullptr) {
            logger = spdlog::basic_logger_mt("mex_logger", "logs/mex.log");
        }
        logger->flush_on(spdlog::level::err);
        //logger->flush_on(spdlog::level::info);
    } catch (const spdlog::spdlog_ex& ex) {
        mexErrMsgIdAndTxt("MATLAB:nearestinterp_cuda:logNoInit", "Log initialization failed: %s", ex.what());
    }

    try {
        logger->info("{:=>50}", " nearestinterp_cuda start");

        /* Check for proper number of input and output arguments */
        if (nrhs != 2) {
            mexErrMsgIdAndTxt( "MATLAB:nearestinterp_cuda:minrhs", "2 input arguments required.");
        } 
        if (nlhs > 2) {
            mexErrMsgIdAndTxt( "MATLAB:nearestinterp_cuda:maxrhs", "Too many output arguments.");
        }

        /* make sure input arguments are expected types */
        if ( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
            mexErrMsgIdAndTxt("MATLAB:nearestinterp_cuda:notDouble", "1st input arg must be type double.");
        }
        if ( !mxIsClass(prhs[1], "int8")) {
            mexErrMsgIdAndTxt("MATLAB:nearestinterp_cuda:notInt8", "2nd input arg must be type int8.");
        }
        if (mxGetNumberOfDimensions(prhs[0]) != 3) {
            mexErrMsgIdAndTxt("MATLAB:nearestinterp_cuda:invalidDim", "# of dimensions of 1st input must be 3.");
        }

        int num_gpus = cudautils::get_gpu_num();
        logger->info("# of gpus = {}", num_gpus);

        double *in_image;
        int8_t *in_map;
        size_t x_size, y_size, z_size;
        double *out_image;

        in_image = mxGetPr(prhs[0]);
        in_map = (int8_t*)mxGetData(prhs[1]);

        const mwSize *image_dims = mxGetDimensions(prhs[0]);
        x_size = image_dims[0];
        y_size = image_dims[1];
        z_size = image_dims[2];

        plhs[0] = mxCreateNumericArray((mwSize)3, image_dims, mxDOUBLE_CLASS, mxREAL);
        out_image = mxGetPr(plhs[0]);

        unsigned int x_sub_size = std::min((unsigned int)2048, (unsigned int)x_size);
        unsigned int y_sub_size = std::min((unsigned int)1024, (unsigned int)y_size);
        unsigned int dx = std::min((unsigned int)256, (unsigned int)x_sub_size);
        unsigned int dy = std::min((unsigned int)256, (unsigned int)y_sub_size);
        const unsigned int dw = 2;

        const unsigned int num_streams = 20;
        logger->info("x_size={},y_size={},z_size={},x_sub_size={},y_sub_size={},dx={},dy={},dw={},# of streams={}",
                x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_streams);

        try {
            cudautils::nearestinterp_bridge(
                    logger, x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams,
                    in_image, in_map, out_image);

        } catch (...) {
            logger->error("internal unknown error occurred");
        }

        logger->info("{:=>50}", " nearestinterp_cuda end");

        logger->flush();
        spdlog::drop_all();
    } catch (...) {
        logger->flush();
        throw;
    }

    return;
}

