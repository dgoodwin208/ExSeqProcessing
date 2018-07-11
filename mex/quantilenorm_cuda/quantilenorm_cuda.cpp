/*=================================================================
 * quantilenorm_cuda.cpp - quantilenorm tif image files
 *
 *  quantilenorm_cuda(datadir, basename, tif_fnames)
 *
 *  datadir(char):  directory to be stored out file
 *  basename(char):  the base name of file names
 *  tif_fnames(cell):  a list of quartets; (tif_fname1, tif_fname2, ..)
 *
 *=================================================================*/
 

#include <thread>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <vector>
#include <set>
#include <tuple>

#include "spdlog/spdlog.h"
#include "mex.h"
#include "mex-utils/tiffs.h"
#include "mex-utils/hdf5.h"
#include "gpudevice.h"
#include "quantilenorm_impl.h"


void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{ 
    std::shared_ptr<spdlog::logger> logger;
    try {
        spdlog::set_async_mode(4096, spdlog::async_overflow_policy::block_retry, nullptr, std::chrono::seconds(2));
        //spdlog::set_async_mode(4096, spdlog::async_overflow_policy::block_retry, nullptr);
        spdlog::set_level(spdlog::level::info);
        //spdlog::set_level(spdlog::level::trace);
        logger = spdlog::get("mex_logger");
        if (logger == nullptr) {
            logger = spdlog::basic_logger_mt("mex_logger", "logs/mex.log");
        }
        logger->flush_on(spdlog::level::err);
        //logger->flush_on(spdlog::level::info);
    } catch (const spdlog::spdlog_ex& ex) {
        mexErrMsgIdAndTxt("MATLAB:quantilenorm_cuda:logNoInit", "Log initialization failed: %s", ex.what());
    }

    try {
        logger->info("{:=>50}", " quantilenorm_cuda start");

        /* Check for proper number of input and output arguments */
        if (nrhs < 3) {
            mexErrMsgIdAndTxt( "MATLAB:quantilenorm_cuda:minrhs", "at least 3 input arguments required.");
        } else if (nrhs > 4) {
            mexErrMsgIdAndTxt( "MATLAB:quantilenorm_cuda:minrhs", "too many input arguments.");
        } 
        if (nlhs > 1) {
            mexErrMsgIdAndTxt( "MATLAB:quantilenorm_cuda:maxrhs", "Too many output arguments.");
        }

        /* make sure input arguments are expected types */
        if ( !mxIsChar(prhs[0])) {
            mexErrMsgIdAndTxt("MATLAB:quantilenorm_cuda:notChar", "1st input arg must be type char.");
        }
        if ( !mxIsChar(prhs[1])) {
            mexErrMsgIdAndTxt("MATLAB:quantilenorm_cuda:notChar", "2nd input arg must be type char.");
        }
        if ( !mxIsCell(prhs[2])) {
            mexErrMsgIdAndTxt("MATLAB:quantilenorm_cuda:notCell", "3rd input arg must be type cell.");
        }
        if (nrhs == 4 && !mxIsLogical(prhs[3])) {
            mexErrMsgIdAndTxt("MATLAB:quantilenorm_cuda:notLogical", "4th input arg must be type logical.");
        }


        int num_gpus = cudautils::get_gpu_num();
        logger->info("# of gpus = {}", num_gpus);


        std::string datadir = std::string(mxArrayToString(prhs[0]));
        logger->info("datadir = {}", datadir.c_str());
        std::string basename = std::string(mxArrayToString(prhs[1]));
        logger->info("basename = {}", basename.c_str());

        const mxArray *root_cell_ptr = prhs[2];
        mwSize total_num_cells = mxGetNumberOfElements(root_cell_ptr);
        if (total_num_cells == 0) {
            mexErrMsgIdAndTxt("MATLAB:quantilenorm_cuda:invalidInput", "Filename is not given.");
        }

        std::vector<std::string> fnames;
        for (int i = 0; i < total_num_cells; i++) {
            const mxArray *elem_ptr = mxGetCell(root_cell_ptr, i);
            if (elem_ptr == NULL) {
                mexErrMsgIdAndTxt("MATLAB:quantilenorm_cuda:invalidInput", "Empty cell.");
            }
            if ( !mxIsChar(elem_ptr)) {
                mexErrMsgIdAndTxt("MATLAB:quantilenorm_cuda:invalidInput", "Invalid input filename.");
            }

            std::string fname(mxArrayToString(elem_ptr));
            fnames.push_back(fname);
            logger->debug("in[{}] = {}", i, fnames[i].c_str());

            if (fname.substr(fname.size() - 4, 4) != ".tif" && fname.substr(fname.size() - 3, 3) != ".h5") {
                mexErrMsgIdAndTxt("MATLAB:quantilenorm_cuda:invalidInput", "Invalid input filetype.");
            }
        }

        size_t image_width, image_height, num_slices;
        bool use_hdf5;
        if (fnames[0].substr(fnames[0].size() - 4, 4) == ".tif") {
            mexutils::gettiffinfo(fnames[0], image_width, image_height, num_slices);
            use_hdf5 = false;
        } else if (fnames[0].substr(fnames[0].size() - 3, 3) == ".h5") {
            mexutils::gethdf5finfo(fnames[0], image_width, image_height, num_slices);
            use_hdf5 = true;
        }

        bool use_tmp_files = true;
        if (nrhs == 4) {
            use_tmp_files = mxIsLogicalScalarTrue(prhs[3]);
        }

        QuantileNormImpl qn_impl(datadir, basename, fnames, image_width, image_height, num_slices, num_gpus, use_hdf5, use_tmp_files);

        try {
            qn_impl.run();
        } catch (ExceptionToMATLAB& ex) {
            mexErrMsgIdAndTxt(ex.getMatlabId().c_str(), ex.getMessage().c_str());
        } catch (...) {
            mexErrMsgIdAndTxt("MATLAB:quantilenorm_impl:unknownError", "internal unknown error occurred");
        }


        if (use_tmp_files) {
            std::vector<std::string> result = qn_impl.getResult();

            plhs[0] = mxCreateCellMatrix(result.size() + 1, 1);
            for (int i = 0; i < result.size(); i++) {
                mxSetCell(plhs[0], i, mxCreateString(result[i].c_str()));
            }
            mxArray *mx_image_size = mxCreateDoubleMatrix(1, 3, mxREAL);
            mxSetCell(plhs[0], result.size(), mx_image_size);

            double *image_size = mxGetPr(mx_image_size);
            image_size[0] = static_cast<double>(image_height);
            image_size[1] = static_cast<double>(image_width);
            image_size[2] = static_cast<double>(num_slices);
        } else {
            std::vector<std::shared_ptr<std::vector<double>>> result = qn_impl.getNormResult();

            plhs[0] = mxCreateCellMatrix(2, 1);

            mxArray *mx_result_mat = mxCreateDoubleMatrix(result[0]->size(), result.size(), mxREAL);
            mxSetCell(plhs[0], 0, mx_result_mat);
            double *out_mat = mxGetPr(mx_result_mat);
            for (int i = 0; i < result.size(); i++) {
                std::shared_ptr<std::vector<double>> array = result[i];
                std::copy(array->data(), array->data() + array->size(), out_mat);
                out_mat += array->size();
            }

            mxArray *mx_image_size = mxCreateDoubleMatrix(1, 3, mxREAL);
            mxSetCell(plhs[0], 1, mx_image_size);

            double *image_size = mxGetPr(mx_image_size);
            image_size[0] = static_cast<double>(image_height);
            image_size[1] = static_cast<double>(image_width);
            image_size[2] = static_cast<double>(num_slices);
        }

        cudautils::resetDevice();

        logger->info("{:=>50}", " quantilenorm_cuda end");

        logger->flush();
        spdlog::drop_all();
    } catch (...) {
        logger->flush();
        throw;
    }

    return;
}

