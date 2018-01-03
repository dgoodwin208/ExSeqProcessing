/*=================================================================
 * semaphore.cpp - interface of posix semaphore
 *
 *  semaphore(name, func)
 *
 *  name(char):  semaphore name
 *  func(char):  semaphore functions; open,post,wait,trywait,close,unlink
 *
 *=================================================================*/
 
#include <string>

#include <semaphore.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "mex.h"


void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{ 
    /* Check for proper number of input and output arguments */
    if (nrhs < 2) {
        mexErrMsgIdAndTxt( "MATLAB:semaphore:minrhs","At least 2 input arguments required.");
    } else if (nrhs > 3) {
        mexErrMsgIdAndTxt( "MATLAB:semaphore:maxrhs","At most 3 input arguments required.");
    }
    if (nlhs > 1) {
        mexErrMsgIdAndTxt( "MATLAB:semaphore:maxrhs","Too many output arguments.");
    }

    /* make sure the first input arguments is type string */
    if ( !mxIsChar(prhs[0])) {
        mexErrMsgIdAndTxt("MATLAB:semaphore:notChar","1st input arg must be type char.");
    }
    if ( !mxIsChar(prhs[1])) {
        mexErrMsgIdAndTxt("MATLAB:semaphore:notChar","2nd input arg must be type char.");
    }

    std::string sem_name = std::string(mxArrayToString(prhs[0]));
//    mexPrintf("sem_name = %s\n", sem_name.c_str());

    std::string func = std::string(mxArrayToString(prhs[1]));
//    mexPrintf("func = %s\n", func.c_str());

    sem_t *sem;
    int ret;
    if (func == "unlink") {
        ret = sem_unlink(sem_name.c_str());
        if (ret == -1) {
            ret = errno;
            mexPrintf("ERR=%d\n", ret);
            mexPrintf("failed to unlink semaphore.\n");
        }
        plhs[0] = mxCreateDoubleScalar(ret);
        return;
    } else if (func == "open") {
        if (nrhs != 3) {
            mexErrMsgIdAndTxt( "MATLAB:semaphore:maxrhs","3 input arguments required.");
        }
        mwSize sem_value = mxGetScalar(prhs[2]);
//        mexPrintf("value = %d\n", sem_value);
        mode_t old_umask = umask(0);
        sem = sem_open(sem_name.c_str(), O_CREAT|O_RDWR, 0777, sem_value);
        umask(old_umask);
        ret = 0;
        if (sem == SEM_FAILED) {
            ret = errno;
            mexPrintf("ERR=%d\n", ret);
            mexPrintf("failed to open semaphore.\n");
        }
        plhs[0] = mxCreateDoubleScalar(ret);
        return;
    }

    sem = sem_open(sem_name.c_str(), O_RDWR);
    ret = errno;
    if (sem == SEM_FAILED) {
        mexPrintf("ERR=%d\n", ret);
        mexPrintf("failed to open semaphore.\n");
        plhs[0] = mxCreateDoubleScalar(ret);
        return;
    }

    if (func == "wait") {
        ret = sem_wait(sem);
    } else if (func == "trywait") {
        ret = sem_trywait(sem);
    } else if (func == "post") {
        ret = sem_post(sem);
    } else if (func == "getvalue") {
        int val;
        ret = sem_getvalue(sem, &val);
        if (ret == 0) {
            ret = val;
        }
    } else if (func == "close") {
        ret = sem_close(sem);
    } else {
        ret = -1;
    }
    plhs[0] = mxCreateDoubleScalar(ret);

    return;
}

