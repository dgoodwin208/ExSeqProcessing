/*=================================================================
 * nnsearch2_cuda.cpp - nearest-neighbor search (k=2)
 *
 *  nnsearch2_cuda(x, y)
 *
 *  Input:
 *    x    : m x k matrix; lists of k-dimensional vectors
 *    y    : n x k matrix; lists of k-dimensional vectors
 *
 *  Output:
 *    r    : m x 2 matrix; lists of 2 tops of minimums
 *    idx  : m x 2 matrix; lists of 2 indices of the 2 tops which show along the n-direction
 *
 *=================================================================*/
 

#include <fstream>
#include <vector>
#include <string>

#include "spdlog/spdlog.h"
#include "nnsearch2.h"
#include "gpudevice.h"
#include "cuda_task_executor.h"

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
        mexErrMsgIdAndTxt("MATLAB:nnsearch2_cuda:logNoInit", "Log initialization failed: %s", ex.what());
    }

    try {
        logger->info("{:=>50}", " nnsearch2_cuda start");

        /* Check for proper number of input and output arguments */
        if (nrhs != 2) {
            mexErrMsgIdAndTxt( "MATLAB:nnsearch2_cuda:minrhs", "2 input arguments required.");
        } 
        if (nlhs > 2) {
            mexErrMsgIdAndTxt( "MATLAB:nnsearch2_cuda:maxrhs", "Too many output arguments.");
        }

        /* make sure input arguments are expected types */
        if ( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
            mexErrMsgIdAndTxt("MATLAB:nnsearch2_cuda:notDouble", "1st input arg must be type double.");
        }
        if ( !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1])) {
            mexErrMsgIdAndTxt("MATLAB:nnsearch2_cuda:notDouble", "2nd input arg must be type double.");
        }


        int num_gpus = cudautils::get_gpu_num();
        logger->info("# of gpus = {}", num_gpus);

        double *inMatrix[2];
        size_t m, n, k, k1;
        double *outMinsVal;
        unsigned int* outMinsIdx;

        inMatrix[0] = mxGetPr(prhs[0]);
        inMatrix[1] = mxGetPr(prhs[1]);

        m = mxGetM(prhs[0]);
        k = mxGetN(prhs[0]);

        n = mxGetM(prhs[1]);
        k1 = mxGetN(prhs[1]);
        if (k != k1) {
            mexErrMsgIdAndTxt("MATLAB:nnsearch2_cuda:invalidInputSize", "Input matrices must be the same column size.");
        }

        plhs[0] = mxCreateDoubleMatrix((mwSize)m, (mwSize)2, mxREAL);
        plhs[1] = mxCreateNumericMatrix((mwSize)m, (mwSize)2, mxUINT32_CLASS, mxREAL);

        outMinsVal = mxGetPr(plhs[0]);
        outMinsIdx = (unsigned int*)mxGetData(plhs[1]);

        unsigned int dm = 1000;
        unsigned int dn = (n > 50000) ? 50000 : n;
        int num_streams = 10;
        //unsigned int num_streams = 1;

        //TODO check the max of GPU memory usage!

        try {
            std::shared_ptr<cudautils::NearestNeighborSearch> nns =
                std::make_shared<cudautils::NearestNeighborSearch>(m, n, k, dm, dn, num_gpus, num_streams);

            cudautils::CudaTaskExecutor executor(num_gpus, num_streams, nns);

            nns->setInput(inMatrix[0], inMatrix[1]);

            executor.run();

            nns->getResult(outMinsVal, outMinsIdx);
        } catch (...) {
            mexErrMsgIdAndTxt("MATLAB:nnsearch2_impl:unknownError", "internal unknown error occurred");
        }

        logger->info("{:=>50}", " nnsearch2_cuda end");

        logger->flush();
        spdlog::drop_all();
    } catch (...) {
        logger->flush();
        throw;
    }

    return;
}

