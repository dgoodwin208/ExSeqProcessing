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
#include "cuda-utils/gpudevice.h"
#include "quantilenorm_impl.h"


void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{ 
    std::shared_ptr<spdlog::logger> logger;
    try {
        spdlog::set_async_mode(4096, spdlog::async_overflow_policy::block_retry, nullptr, std::chrono::seconds(2));
        //spdlog::set_async_mode(4096, spdlog::async_overflow_policy::block_retry, nullptr);
        spdlog::set_level(spdlog::level::trace);
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
        if (nrhs != 3) {
            mexErrMsgIdAndTxt( "MATLAB:quantilenorm_cuda:minrhs", "3 input arguments required.");
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


        int num_gpus = cudautils::get_gpu_num();
        logger->info("# of gpus = {}", num_gpus);


        std::string datadir = std::string(mxArrayToString(prhs[0]));
        logger->info("datadir = {}", datadir.c_str());
        std::string basename = std::string(mxArrayToString(prhs[1]));
        logger->info("basename = {}", basename.c_str());

        const mxArray *root_cell_ptr = prhs[2];
        mwSize total_num_cells = mxGetNumberOfElements(root_cell_ptr);
//        logger->debug("total_num_cells = {}", total_num_cells);

        std::vector<std::string> tif_fnames;
        for (int i = 0; i < total_num_cells; i++) {
            const mxArray *elem_ptr = mxGetCell(root_cell_ptr, i);
            if (elem_ptr == NULL) {
                mexErrMsgIdAndTxt("MATLAB:quantilenorm_cuda:invalidInput", "Empty cell.");
            }
            if ( !mxIsChar(elem_ptr)) {
                mexErrMsgIdAndTxt("MATLAB:quantilenorm_cuda:invalidInput", "Invalid input filename.");
            }

            std::string fname(mxArrayToString(elem_ptr));
            tif_fnames.push_back(fname);
            logger->debug("in[{}] = {}", i, tif_fnames[i].c_str());
        }

        size_t image_width, image_height, num_slices;
        mexutils::gettiffinfo(tif_fnames[0], image_width, image_height, num_slices);

        QuantileNormImpl qn_impl(datadir, basename, tif_fnames, image_width, image_height, num_slices, num_gpus);

        try {
            qn_impl.run();
        } catch (ExceptionToMATLAB& ex) {
            mexErrMsgIdAndTxt(ex.getMatlabId().c_str(), ex.getMessage().c_str());
        } catch (...) {
            mexErrMsgIdAndTxt("MATLAB:quantilenorm_impl:unknownError", "internal unknown error occurred");
        }

        std::vector<std::string> result = qn_impl.getResult();

        plhs[0] = mxCreateCellMatrix(result.size() + 2, 1);
        for (int i = 0; i < result.size(); i++) {
            mxSetCell(plhs[0], i, mxCreateString(result[i].c_str()));
        }
        mxSetCell(plhs[0], result.size()    , mxCreateDoubleScalar((double)image_height));
        mxSetCell(plhs[0], result.size() + 1, mxCreateDoubleScalar((double)image_width));

        logger->info("{:=>50}", " quantilenorm_cuda end");

        logger->flush();
        spdlog::drop_all();
    } catch (...) {
        logger->flush();
        throw;
    }

    return;
}

