#include "mex.h"
/* #include "cudnn-utils/conv_sample.h" */
#include "cufft-utils/cufftutils.h"

/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    double *inArray1;
    double *inArray2;
    float *outArray;
    size_t mrows1, ncols1;
    size_t mrows2, ncols2;
    size_t inArraySize;
    int benchmark = 1; //time it
    int column_order = 0; // data from MATLAB must be treated as column-order

    /* Check for proper number of input and output arguments */    
    //if (nrhs != 2) {
        //mexErrMsgIdAndTxt( "convn_cuda:InvalidInput","Two input arguments required.");
    //} 
    if (nlhs > 1){
        mexErrMsgIdAndTxt( "convn_cuda:InvalidOutput","Only 0 or 1 output arguments.");
    }

    /* make sure input array argument is type single */
    if ( !mxIsSingle(prhs[0])) {
        mexErrMsgIdAndTxt("convn_cuda:InvalidInput","Input array must be type single (float).");
    }

    // Handle image and filter array sizes
    int *image_size;
    const mwSize image_dims = mxGetNumberOfDimensions(prhs[0]);
    image_size = (int *) mxGetDimensions(prhs[0]);
    if (benchmark)
        printf("image size: %d, %d, %d\n", image_size[0], image_size[1], image_size[2]);

    int lengths[3];
    lengths[0] = (int)* mxGetPr(prhs[1]);
    lengths[1] = (int)* mxGetPr(prhs[2]);
    lengths[2] = (int)* mxGetPr(prhs[3]);
    if (benchmark)
        printf("filter lengths: %d, %d, %d\n", lengths[0], lengths[1], lengths[2]);

    // create a new single real matrix on the heap to place output data on (wastes memory but is more usable than in-place)
    plhs[0] = mxCreateNumericArray(image_dims, (mwSize* ) image_size, mxSINGLE_CLASS, mxREAL);
    outArray = (float *) mxGetData(plhs[0]);

    // generate params
    /* create a pointer to the real data in the input array,  */
    float *image = (float *) mxGetData(prhs[0]); // GetData returns type void *
    cufftutils::fft3(image, image_size, lengths, outArray, column_order);

}

