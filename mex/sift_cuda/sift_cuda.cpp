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

#include "gpulock.h"
#include "gpudevice.h"
#include "sift_bridge.h"

#include "spdlog/spdlog.h"

#include "mex.h"
#include "matrix.h"

#include "mexutil.h"
//#include "sift_types.h"
#include <stdio.h>

/* Convert from a Keypoint_store to an array of MATLAB keypoint structs. 
 * Returns the array of keypoints, or NULL on failure. */
mxArray *kp2mx(cudautils::Keypoint_store * kp, 
        cudautils::SiftParams sift_params) {

    mxArray *mxKp;
    double xyScale, tScale;
    uint8_t* ivec;
    int xNum, yNum, zNum, tscaleNum, xyscaleNum, ivecNum;

    /* Keypoint struct information */
    const char *fieldNames[] = {
            X_NAME,
            Y_NAME,
            Z_NAME,
            TSCALE_NAME,
            XYSCALE_NAME,
            IVEC_NAME,
    };
    const mwSize numKp = (mwSize) kp->len;

    const mwSize kpNDims = 1;
    const int kpNFields = sizeof(fieldNames) / sizeof(char *);
	// Make an array of structs for the output 
	mxKp = mxCreateStructArray(kpNDims, &numKp, kpNFields, 
                                            fieldNames);
        if (mxKp == NULL)
                return NULL;

        // Get the field indices in the structs
        if (    (xNum = mxGetFieldNumber(mxKp, X_NAME)) < 0 ||
                (yNum = mxGetFieldNumber(mxKp, Y_NAME)) < 0 ||
                (zNum = mxGetFieldNumber(mxKp, Z_NAME)) < 0 ||
                (tscaleNum = mxGetFieldNumber(mxKp, TSCALE_NAME)) < 0 ||
                (xyscaleNum = mxGetFieldNumber(mxKp, XYSCALE_NAME)) < 0 ||
                (ivecNum = mxGetFieldNumber(mxKp, IVEC_NAME)) < 0 )
                return NULL;

        // Write the keypoints to the output
        for (int i = 0; i < numKp; i++) {

                mxArray *x, *y, *z, *mxtScale, *mxxyScale, *mxIvec;
                uint8_t *ivec;

                cudautils::Keypoint *const key = kp->buf + i;

                if (!sift_params.skipDescriptor) {
                    // Initialize the ivec array
                    const mwSize dims[2] = {1, (int) sift_params.descriptor_len};
                    if ((mxIvec = 
                            mxCreateNumericArray(2, dims,
                                mxUINT8_CLASS, mxREAL)) == NULL)
                            return NULL;

                    ivec = (uint8_t *) mxGetData(mxIvec); 
                    for (int j = 0; j < sift_params.descriptor_len; j++) 
                        ivec[j] = (uint8_t) key->ivec[j];

                    // Copy the scale 
                    mxtScale = mxCreateDoubleScalar(key->tScale);
                    mxxyScale = mxCreateDoubleScalar(key->xyScale);

                    // Set the descriptor struct fields
                    mxSetFieldByNumber(mxKp, i, tscaleNum, mxtScale);
                    mxSetFieldByNumber(mxKp, i, xyscaleNum, mxxyScale);
                    mxSetFieldByNumber(mxKp, i, ivecNum, mxIvec);
                }

                x = mxCreateDoubleScalar(key->x);
                y = mxCreateDoubleScalar(key->y);
                z = mxCreateDoubleScalar(key->z);

                // Set the struct fields
                mxSetFieldByNumber(mxKp, i, xNum, x);
                mxSetFieldByNumber(mxKp, i, yNum, y);
                mxSetFieldByNumber(mxKp, i, zNum, z);
        }

        return mxKp;
}

bool get_logical_field(const mxArray* prhs[], const char* name) {
    mxArray* tmp = mxGetField(prhs[2], 0, name);
    if ((tmp != NULL) &&
        (mxGetClassID(tmp) == mxLOGICAL_CLASS)) {
        return *((bool*) mxGetPr(tmp));
    } else {
        mexErrMsgIdAndTxt("MATLAB:sift_cuda:sift_params", "sift_params missing field: %s", name);
    }
}

double* get_double_ptr_field(const mxArray* prhs[], const char* name) {
    mxArray* tmp = mxGetField(prhs[2], 0, name);
    if ((tmp != NULL) &&
        (mxGetClassID(tmp) == mxDOUBLE_CLASS)) {
        return mxGetPr(tmp);
    } else {
        mexErrMsgIdAndTxt("MATLAB:sift_cuda:sift_params", "sift_params missing field: %s", name);
    }
}

double get_double_field(const mxArray* prhs[], const char* name) {
    mxArray* tmp = mxGetField(prhs[2], 0, name);
    if ((tmp != NULL) &&
        (mxGetClassID(tmp) == mxDOUBLE_CLASS)) {
        return *((double*) mxGetPr(tmp));
    } else {
        mexErrMsgIdAndTxt("MATLAB:sift_cuda:sift_params", "sift_params missing field: %s", name);
    }
}

// parse SiftParams struct
cudautils::SiftParams get_params(const mxArray* prhs[]) {

    cudautils::SiftParams sift_params;

    sift_params.MagFactor = get_double_field(prhs, "MagFactor");

    sift_params.TwoPeak_Thresh = get_double_field(prhs, "TwoPeak_Thresh");

    sift_params.IndexSigma = get_double_field(prhs, "IndexSigma");

    sift_params.IndexSize = get_double_field(prhs, "IndexSize");

    sift_params.nFaces = (int) get_double_field(prhs, "nFaces");

    sift_params.Tessellation_levels = get_double_field(prhs, "Tessellation_levels");

    sift_params.Smooth_Flag = get_logical_field(prhs, "Smooth_Flag");

    sift_params.skipDescriptor = get_logical_field(prhs, "skipDescriptor");

    sift_params.Smooth_Var = get_double_field(prhs, "Smooth_Var");

    sift_params.SigmaScaled = get_double_field(prhs, "SigmaScaled");

    sift_params.Tessel_thresh = get_double_field(prhs, "Tessel_thresh");

    sift_params.TwoPeak_Flag = get_logical_field(prhs, "TwoPeak_Flag");

    sift_params.xyScale = get_double_field(prhs, "xyScale");

    sift_params.tScale = get_double_field(prhs, "tScale");

    sift_params.MaxIndexVal = get_double_field(prhs, "MaxIndexVal");

    sift_params.keypoint_num = (int) get_double_field(prhs, "keypoint_num");

    sift_params.fv_centers_len = (int) get_double_field(prhs, "fv_centers_len");

    const mwSize *image_dims = mxGetDimensions(prhs[0]);
    sift_params.image_size0 = image_dims[0];
    sift_params.image_size1 = image_dims[1];
    sift_params.image_size2 = image_dims[2];

    sift_params.descriptor_len = (int) get_double_field(prhs, "descriptor_len");

    return sift_params;
}

void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]) { 
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
        double* fv_centers = get_double_ptr_field(prhs, "fv_centers");
        int stream_num = (int) get_double_field(prhs, "stream_num");
        int x_substream_stride = (int) get_double_field(prhs, "x_substream_stride");
        int y_substream_stride = (int) get_double_field(prhs, "y_substream_stride");

        int num_gpus = cudautils::get_gpu_num();
        logger->info("# of gpus = {}", num_gpus);

        cudautils::Keypoint_store keystore;

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

        // size to assign each GPU
        unsigned int x_sub_size = x_size;
        unsigned int y_sub_size = (unsigned int) (y_size + num_gpus - 1) / num_gpus;

        // stride must be <= GPU data size
        if (x_substream_stride > x_sub_size) {
            x_substream_stride =  x_sub_size;
            stream_num = 1;
            mexWarnMsgIdAndTxt("MATLAB:sift_cuda:invalidValue", "x_substream_stride can not excede the image x dimension size\n\tSwitched x_substream_stride");
        }

        if (y_substream_stride > y_sub_size ) {
            y_substream_stride =  y_sub_size;
            stream_num = 1;
            mexWarnMsgIdAndTxt("MATLAB:sift_cuda:invalidValue", "y_substream_stride can not excede the image ydimension size / # of GPUs\n\tSwitched y_substream_stride");
        }

        // padding for SIFT
        const unsigned int orihist_radius = static_cast<unsigned int>(sift_params.xyScale * 3.0);
        const unsigned int xyiradius = static_cast<unsigned int>(1.414 * sift_params.xyScale * sift_params.MagFactor * (sift_params.IndexSize + 1) / 2.0);
        unsigned int dw = (num_gpus == 1 ? 0 : std::max(orihist_radius, xyiradius) + 1);
        if (dw > y_substream_stride) {
            if (dw > y_size) {
                num_gpus = 1;
                stream_num = 1;
                dw = 0;
                y_sub_size = y_size;
                y_substream_stride = y_sub_size;
                logger->info("dw is larger than y_size. Set num_gpus = 1 and stream_num = 1");
                mexWarnMsgIdAndTxt("MATLAB:sift_cuda:invalidValue", "dw can not excede y_size\n\tSwitched to num_gpus=1 and stream_num=1");
            } else {
                if (stream_num > 1) {
                    mexErrMsgIdAndTxt("MATLAB:sift_cuda:invalidValue", "y_substream_stride is too small in multi-GPUs.\n\tPlease set smaller stream_num in sift_params");
                }
                mexErrMsgIdAndTxt("MATLAB:sift_cuda:invalidValue", "y_size is too small in this GPU configuration.\n\tPlease use larger size of input image in sift_params");
            }
        }


        logger->info("x_size={},y_size={},z_size={},x_sub_size={},y_sub_size={},x_substream_stride={},y_substream_stride={}, dw={}, # of streams={}",
                x_size, y_size, z_size, x_sub_size, y_sub_size,
                x_substream_stride, y_substream_stride, dw, stream_num);

        cudautils::GPULock lock(num_gpus);
        lock.trylockall();
        try {
            cudautils::sift_bridge(
                    logger, x_size, y_size, z_size, x_sub_size, y_sub_size, x_substream_stride,
                    y_substream_stride, dw, num_gpus, stream_num, in_image, in_map,
                    sift_params, fv_centers, &keystore);

        } catch (...) {
            logger->error("internal unknown error occurred");
        }
        lock.unlockall();
        logger->info("Sift_bridge completed");

        // Convert the output keypoints
        if (keystore.len) {
            logger->info("Converting keypoints");
            if ((plhs[0] = kp2mx(&keystore, sift_params)) == NULL)
                logger->error("keystore to mex error occurred");
        } else {
            logger->info("All keypoints removed, returning empty cell");
            plhs[0] = mxCreateDoubleMatrix(0,0, mxREAL);
        }
        logger->info("Outputs assigned successfully");

        logger->info("{:=>50}", " sift_cuda end");

        logger->flush();
        spdlog::drop_all();
    } catch (...) {
        logger->flush();
        throw;
    }

    return;
}

