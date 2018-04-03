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

    /* Check for proper number of input and output arguments */    
    if (nrhs < 2) {
        mexErrMsgIdAndTxt( "convn_cuda:InvalidInput","Two input arguments required.");
    } 
    if (nlhs > 1){
        mexErrMsgIdAndTxt( "convn_cuda:InvalidOutput","Only 0 or 1 output arguments.");
    }

    int benchmark = 0; //time it, print debugging info
    if (nrhs > 2) 
        benchmark = (int)* mxGetPr(prhs[2]);

    /* make sure input array arguments are type single */
    if ( !mxIsSingle(prhs[0])) {
        mexErrMsgIdAndTxt("convn_cuda:InvalidInput","Input array must be type single (float).");
    }
    if ( !mxIsSingle(prhs[1])) {
        mexErrMsgIdAndTxt("convn_cuda:InvalidInput","Input array must be type single (float).");
    }

    // Handle image and filter array sizes
    int *image_size;
    const mwSize image_dims = mxGetNumberOfDimensions(prhs[0]);
    image_size = (int *) mxGetDimensions(prhs[0]);
    if (benchmark)
        printf("image size: %d, %d, %d\n", image_size[0], image_size[1], image_size[2]);

    //const size_t *filter_size;
    int *filter_size;
    const mwSize filter_dims = mxGetNumberOfDimensions(prhs[1]);
    filter_size = (int *) mxGetDimensions(prhs[1]);
    if (benchmark)
        printf("filter size: %d, %d, %d\n", filter_size[0], filter_size[1], filter_size[2]);

    // create a new single real matrix on the heap to place output data on (wastes memory but is more usable than in-place)
    plhs[0] = mxCreateNumericArray(image_dims, (mwSize* ) image_size, mxSINGLE_CLASS, mxREAL);
    outArray = (float *) mxGetData(plhs[0]);

    // generate params
    int algo = 0; // forward convolve
    bool column_order = true; // bool for 
    // pading the image to m + n -1 per dimension
    /* create a pointer to the real data in the input array,  */
    float *image = (float *) mxGetData(prhs[0]); // GetData returns type void *
    float *filter = (float *) mxGetData(prhs[1]); // GetData returns type void *
    cufftutils::conv_1GPU_handler(image, filter, outArray, algo, image_size, filter_size, column_order, benchmark);
}
