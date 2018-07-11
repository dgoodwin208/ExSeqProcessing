/*=================================================================
 * quantilenorm_cuda_final.cpp - finalize for quantilenorm_cuda
 *
 *  quantilenorm_cuda_final(num_gpu_sem, num_core_sem)
 *
 *  num_gpu_sem(scalar):  # semaphores for gpus
 *  num_core_sem(scalar):  # semaphores for cores
 *
 *=================================================================*/

#include <semaphore.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "mex.h"
#include "spdlog/spdlog.h"

void
finalize_semaphores(const int num_sems, const std::string& prefix) {
    auto logger = spdlog::get("mex_logger");

    logger->debug("num_sems = {}", num_sems);

    int ret = 0;
    for (size_t i = 0; i < num_sems; i++) {
        std::string sem_name = prefix + std::to_string(i);
        logger->debug("finalize sem: {}", sem_name.c_str());
        ret = sem_unlink(sem_name.c_str());
        int errn = errno;
        if (ret == -1) {
            mexErrMsgIdAndTxt("MATLAB:quantilenorm_cuda_final:semaphoreNotUnlinked", "failed to unlink semaphore %s. ERR=%d", sem_name.c_str(), errn);
        }
    }
}

void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{ 
    try {
        spdlog::set_level(spdlog::level::trace);
        auto logger = spdlog::get("mex_logger");
        if (logger == nullptr) {
            spdlog::set_async_mode(4096);
            logger = spdlog::basic_logger_mt("mex_logger", "logs/mex.log");
        }
        logger->info("{:=>50}", " quantilenorm_cuda_final");

        /* Check for proper number of input and output arguments */
        if (nrhs < 1) {
            mexErrMsgIdAndTxt( "MATLAB:quantilenorm_cuda_final:minrhs", "at least 1 input argument required.");
        } else if (nrhs > 2) {
            mexErrMsgIdAndTxt( "MATLAB:quantilenorm_cuda_final:maxrhs", "too many input arguments.");
        } 
        if (nlhs > 0) {
            mexErrMsgIdAndTxt( "MATLAB:quantilenorm_cuda_final:maxrhs", "Too many output arguments.");
        }

        /* make sure the first and second input arguments are type scalar */
        if ( !mxIsScalar(prhs[0])) {
            mexErrMsgIdAndTxt("MATLAB:quantilenorm_cuda_final:notScalar","Input must be type scalar.");
        }
        if (nrhs == 2 && !mxIsScalar(prhs[1])) {
            mexErrMsgIdAndTxt("MATLAB:quantilenorm_cuda_final:notScalar","Input must be type scalar.");
        }

        int num_gpu_sem = (int)mxGetScalar(prhs[0]);
        finalize_semaphores(num_gpu_sem, "/g");

        if (nrhs == 2) {
            int num_core_sem = (int)mxGetScalar(prhs[1]);
            finalize_semaphores(num_core_sem, "/qn_c");
        }

        logger->flush();
        spdlog::drop_all();
    } catch (const spdlog::spdlog_ex& ex) {
        mexErrMsgIdAndTxt("MATLAB:quantilenorm_cuda_final:logNoInit", "Log initialization failed: %s", ex.what());
    }

    return;
}

