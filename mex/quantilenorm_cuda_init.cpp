/*=================================================================
 * quantilenorm_cuda_init.cpp - initialize for quantilenorm_cuda
 *
 *  quantilenorm_cuda(num_gpu_sem, num_core_sem)
 *
 *  num_gpu_sem(cell):  a list of pairs of # gpus and # semaphores
 *  num_core_sem(cell):  a list of pairs of # cores and # semaphores
 *
 *=================================================================*/

#include <semaphore.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "mex.h"
#include "spdlog/spdlog.h"

void
init_semaphores(const mxArray *elem_ptr, const std::string& prefix) {
    auto logger = spdlog::get("mex_logger");

    mwSize num_cells = mxGetNumberOfElements(elem_ptr);
    logger->debug("num_cells = {}", num_cells);

    double *val = mxGetPr(elem_ptr);

    for (size_t i = 0; i < num_cells; i++) {
        size_t num_sem_values = (size_t)val[i];

        sem_t *sem;
        std::string sem_name = prefix + std::to_string(i);
        sem_unlink(sem_name.c_str());
        mode_t old_umask = umask(0);
        sem = sem_open(sem_name.c_str(), O_CREAT|O_RDWR, 0777, num_sem_values);
        umask(old_umask);
        logger->debug("init sem: {} ({})", sem_name.c_str(), num_sem_values);
        int ret = errno;
        if (sem == SEM_FAILED) {
            mexErrMsgIdAndTxt("MATLAB::quantilenorm_cuda_init::semaphoreNotCreated", "failed to create semaphore %s. ERR=%d", sem_name.c_str(), ret);
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
        logger->info("{:=>50}", " quantilenorm_cuda_init");

        /* Check for proper number of input and output arguments */
        if (nrhs != 2) {
            mexErrMsgIdAndTxt( "MATLAB:quantilenorm_cuda_init:minrhs", "2 input arguments required.");
        } 
        if (nlhs > 0) {
            mexErrMsgIdAndTxt( "MATLAB:quantilenorm_cuda_init:maxrhs", "Too many output arguments.");
        }

        /* make sure the first and second input arguments are type double */
        if ( !mxIsDouble(prhs[0]) || 
              mxIsComplex(prhs[0])) {
            mexErrMsgIdAndTxt("MATLAB:quantilenorm_cuda_init:notDouble","Input matrix must be type double.");
        }
        if ( !mxIsDouble(prhs[1]) || 
              mxIsComplex(prhs[1])) {
            mexErrMsgIdAndTxt("MATLAB:quantilenorm_cuda_init:notDouble","Input matrix must be type double.");
        }

        const mxArray *gpu_ptr = prhs[0];
        init_semaphores(gpu_ptr, "/g");

        const mxArray *core_ptr = prhs[1];
        init_semaphores(core_ptr, "/qn_c");

        logger->flush();
        spdlog::drop_all();
    } catch (const spdlog::spdlog_ex& ex) {
        mexErrMsgIdAndTxt("MATLAB:quantilenorm_cuda_init:logNoInit", "Log initialization failed: %s", ex.what());
    }

    return;
}

