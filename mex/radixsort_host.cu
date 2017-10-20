#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/system/tbb/execution_policy.h>

#include "mex.h"
#include "gpu/mxGPUArray.h"

/*
 * Device code
 */



/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    double *inArray1;
    double *inArray2;
    double *outArray;
    size_t mrows1, ncols1;
    size_t mrows2, ncols2;
    size_t inArraySize;

    cudaGetLastError();
//    cudaDeviceReset();

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Check for proper number of input and output arguments */    
    if (nrhs != 2) {
        mexErrMsgIdAndTxt( "parallel:gpu:radixsort:InvalidInput","Two input arguments required.");
    } 
    if (nlhs > 1){
        mexErrMsgIdAndTxt( "parallel:gpu:radixsort:InvalidOutput","Many output arguments.");
    }

    /* make sure input array arguments are type double */
    if ( !mxIsDouble(prhs[0]) || 
          mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("parallel:gpu:radixsort:InvalidInput","Input array must be type double.");
    }
    if ( !mxIsDouble(prhs[1]) || 
          mxIsComplex(prhs[1])) {
        mexErrMsgIdAndTxt("parallel:gpu:radixsort:InvalidInput","Input array must be type double.");
    }


    /* create a pointer to the real data in the input array  */
    inArray1 = mxGetPr(prhs[0]);
    inArray2 = mxGetPr(prhs[1]);

    /* get dimensions of the input array */
    mrows1 = mxGetM(prhs[0]);
    ncols1 = mxGetN(prhs[0]);
    if (ncols1 != 1) {
        mexErrMsgIdAndTxt("parallel:gpu:radixsort:InvalidInput","Input array must be M x 1.");
    }
    inArraySize = mrows1;

    mrows2 = mxGetM(prhs[1]);
    ncols2 = mxGetN(prhs[1]);
    if (ncols2 != 1) {
        mexErrMsgIdAndTxt("parallel:gpu:radixsort:InvalidInput","Input array must be M x 1.");
    }
    if (mrows1 != mrows2) {
        mexErrMsgIdAndTxt("parallel:gpu:radixsort:InvalidInput","Input arrays' size is different.");
    }

    plhs[0] = mxCreateDoubleMatrix((mwSize)inArraySize,(mwSize)2,mxREAL);

    outArray = mxGetPr(plhs[0]);

    // Copy data onto the GPU
    thrust::device_vector<double> d_keys(inArray1, inArray1+inArraySize);
    thrust::device_vector<double> d_values(inArray2, inArray2+inArraySize);

    thrust::sort_by_key(thrust::tbb::par, d_keys.begin(), d_keys.end(), d_values.begin());

    thrust::copy(d_keys.begin(), d_keys.end(), outArray);
    thrust::copy(d_values.begin(), d_values.end(), outArray+inArraySize);

}

