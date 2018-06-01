/*=================================================================
 * sift_cuda.cpp - interpolate volume image data from the nearest intensities
 *
 *  sift_cuda(vol_image, map)
 *
 *  Input:
 *    vol_image:  volume image data
 *    map:        mask map data (1: mask, 0: hole)
 *
 *  Output:
 *
 *=================================================================*/
 

#include <fstream>
#include <vector>
#include <algorithm>

#include "gpudevice.h"
#include "sift_bridge.h"

#include "spdlog/spdlog.h"

#include "mex.h"
#include "matrix.h"

#include "sift_types.h"
#include <stdio.h>

/* Convert from a Keypoint_store to an array of MATLAB keypoint structs. 
 * Returns the array of keypoints, or NULL on failure. */
/*
mxArray *kp2mx(cudautils::Keypoint_store * kp) {

    mxArray *mxKp;
    int i, x, y, z;
    double xyScale, tScale;
    double* ivec;

    const mwSize numKp = (mwSize) kp->len;

	// Make an array of structs for the output 
	mxKp = mxCreateStructArray(kpNDims, &numKp, kpNFields, 
                                            fieldNames);
        if (mxKp == NULL)
                return NULL;

        // Get the field indices in the structs
        if ((coordsNum = mxGetFieldNumber(mxKp, COORDS_NAME)) < 0 ||
                (scaleNum = mxGetFieldNumber(mxKp, SCALE_NAME)) < 0 ||
                (oriNum = mxGetFieldNumber(mxKp, ORI_NAME)) < 0 ||
                (octaveNum = mxGetFieldNumber(mxKp, OCTAVE_NAME)) < 0 || 
                (levelNum = mxGetFieldNumber(mxKp, LEVEL_NAME)) < 0)
                return NULL;

        // Write the keypoints to the output
        for (i = 0; i < numKp; i++) {

                mxArray *mxCoords, *mxScale, *mxOri, *mxOctave, *mxLevel;
                double *coords;

                cudautils::Keypoint *const key = kp->buf + i;

                // Initialize the coordinate array
                if ((mxCoords = 
                        mxCreateDoubleMatrix(1, IM_NDIMS, mxREAL)) == NULL)
                        return NULL;

                // Copy the coordinates 
                coords = mxGetData(mxCoords); 
                coords[0] = key->xd;
                coords[1] = key->yd;
                coords[2] = key->zd;

                // Copy the transposed orientation
                if ((mxOri = mat2mx(&key->R)) == NULL)
                        return NULL;

                // Copy the scale 
                mxScale = mxCreateDoubleScalar(key->sd);

                // Copy the octave index
                mxOctave = mxCreateDoubleScalar((double) key->o); 

                // Copy the level index
                mxLevel = mxCreateDoubleScalar((double) key->s);
                
                // Set the struct fields
                mxSetFieldByNumber(mxKp, i, coordsNum, mxCoords);
                mxSetFieldByNumber(mxKp, i, scaleNum, mxScale);
                mxSetFieldByNumber(mxKp, i, oriNum, mxOri);
                mxSetFieldByNumber(mxKp, i, octaveNum, mxOctave);
                mxSetFieldByNumber(mxKp, i, levelNum, mxLevel);
        }

        return mxKp;
}
*/

void init_Keypoint_store(cudautils::Keypoint_store * kp, cudautils::SiftParams sift_params) {
    kp->len = sift_params.keypoint_num;
    kp->buf = (cudautils::Keypoint*) malloc(kp->len * sift_params.descriptor_len * sizeof(double));
}

// parse SiftParams struct
cudautils::SiftParams get_params(const mxArray* prhs[]) {

    cudautils::SiftParams sift_params;
    mxArray* tmp;

    tmp = mxGetField(prhs[2], 0, "MagFactor");
    sift_params.MagFactor = *((double*) mxGetPr(tmp));

    tmp = mxGetField(prhs[2], 0, "IndexSize");
    sift_params.IndexSize = *((int*) mxGetPr(tmp));

    tmp = mxGetField(prhs[2], 0, "nFaces");
    sift_params.nFaces = *((int*) mxGetPr(tmp));

    tmp = mxGetField(prhs[2], 0, "Tessellation_levels");
    sift_params.Tessellation_levels = *((int*) mxGetPr(tmp));

    tmp = mxGetField(prhs[2], 0, "Smooth_Flag");
    sift_params.Smooth_Flag = *((int*) mxGetPr(tmp));

    tmp = mxGetField(prhs[2], 0, "SigmaScaled");
    sift_params.SigmaScaled = *((double*) mxGetPr(tmp));

    tmp = mxGetField(prhs[2], 0, "Tessel_thresh");
    sift_params.Tessel_thresh = *((double*) mxGetPr(tmp));

    tmp = mxGetField(prhs[2], 0, "TwoPeak_Flag");
    sift_params.TwoPeak_Flag = *((int*) mxGetPr(tmp));

    tmp = mxGetField(prhs[2], 0, "xyScale");
    sift_params.xyScale = *((double*) mxGetPr(tmp));

    tmp = mxGetField(prhs[2], 0, "tScale");
    sift_params.tScale = *((double*) mxGetPr(tmp));

    tmp = mxGetField(prhs[2], 0, "MaxIndexVal");
    sift_params.MaxIndexVal = *((double*) mxGetPr(tmp));

    tmp = mxGetField(prhs[2], 0, "fv_centers");
    //FIXME cudaMalloc
    sift_params.fv_centers = *((double**) mxGetPr(tmp));

    tmp = mxGetField(prhs[2], 0, "fv_centers_len");
    sift_params.fv_centers_len = *((int*) mxGetPr(tmp));

    const mwSize *image_dims = mxGetDimensions(prhs[0]);
    //FIXME cudaMalloc
    sift_params.image_size0 = image_dims[0];
    sift_params.image_size1 = image_dims[1];
    sift_params.image_size2 = image_dims[2];

    tmp = mxGetField(prhs[2], 0, "keypoint_num");
    sift_params.keypoint_num = *((int*) mxGetPr(tmp));

    tmp = mxGetField(prhs[2], 0, "descriptor_len");
    sift_params.descriptor_len = *((int*) mxGetPr(tmp));

    return sift_params;
}


void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{ 
    //
    //SIFT3D_Descriptor_store desc;
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
        mexErrMsgIdAndTxt("MATLAB:sift_cuda:logNoInit", "Log initialization failed: %s", ex.what());
    }

    try {
        logger->info("{:=>50}", " sift_cuda start");

        /* Check for proper number of input and output arguments */
        if (nrhs != 3) {
            mexErrMsgIdAndTxt( "MATLAB:sift_cuda:minrhs", "3 input arguments required.");
        } 
        if (nlhs > 2) {
            mexErrMsgIdAndTxt( "MATLAB:sift_cuda:maxrhs", "Too many output arguments.");
        }

        /* make sure input arguments are expected types */
        if ( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
            mexErrMsgIdAndTxt("MATLAB:sift_cuda:notDouble", "1st input arg must be type double.");
        }
        if ( !mxIsClass(prhs[1], "int8")) {
            mexErrMsgIdAndTxt("MATLAB:sift_cuda:notInt8", "2nd input arg must be type int8.");
        }
        if ( !mxIsStruct(prhs[2])) {
            mexErrMsgIdAndTxt("MATLAB:sift_cuda:inputNotStruct", "3rd input arg must be type struct.");
        }
        if (mxGetNumberOfElements(prhs[2]) != 1) {
            mexErrMsgIdAndTxt("MATLAB:sift_cuda:structArray", "3rd input arg of struct must be 1 element long.");
        }
        if (mxGetNumberOfDimensions(prhs[0]) != 3) {
            mexErrMsgIdAndTxt("MATLAB:sift_cuda:invalidDim", "# of dimensions of 1st input must be 3.");
        }

        cudautils::SiftParams sift_params = get_params(prhs);
        cudautils::Keypoint_store keystore;
        int num_gpus = cudautils::get_gpu_num();
        logger->info("# of gpus = {}", num_gpus);

        init_Keypoint_store(&keystore, sift_params); //host alloc mem for num kypts

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
            cudautils::sift_bridge(
                    logger, x_size, y_size, z_size, x_sub_size, y_sub_size, dx, dy, dw, num_gpus, num_streams,
                    in_image, in_map, sift_params, keystore);

        } catch (...) {
            logger->error("internal unknown error occurred");
        }

        logger->info("{:=>50}", " sift_cuda end");

        logger->flush();
        spdlog::drop_all();
    } catch (...) {
        logger->flush();
        throw;
    }

    return;
}

