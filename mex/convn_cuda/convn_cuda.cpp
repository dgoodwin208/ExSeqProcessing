#include "mex.h"
#include "cudnn-utils/conv_sample.h"

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
    if (nrhs != 2) {
        mexErrMsgIdAndTxt( "convn_cuda:InvalidInput","Two input arguments required.");
    } 
    if (nlhs > 1){
        mexErrMsgIdAndTxt( "convn_cuda:InvalidOutput","Only 0 or 1 output arguments.");
    }

    //[> make sure input array arguments are type double <]
    //if ( !mxIsDouble(prhs[0]) || 
          //mxIsComplex(prhs[0])) {
        //mexErrMsgIdAndTxt("convn_cuda:InvalidInput","Input array must be type double.");
    //}
    //if ( !mxIsDouble(prhs[1]) || 
          //mxIsComplex(prhs[1])) {
        //mexErrMsgIdAndTxt("convn_cuda:InvalidInput","Input array must be type double.");
    //}

    // Handle image and filter array sizes
    int *image_size;
    const mwSize image_dims = mxGetNumberOfDimensions(prhs[0]);
    image_size = (int *) mxGetDimensions(prhs[0]);
    printf("image size: %d, %d, %d\n", image_size[0], image_size[1], image_size[2]);

    //const size_t *filter_size;
    int *filter_size;
    const mwSize filter_dims = mxGetNumberOfDimensions(prhs[1]);
    filter_size = (int *) mxGetDimensions(prhs[1]);
    printf("filter size: %d, %d, %d\n", filter_size[0], filter_size[1], filter_size[2]);

    //if (ncols1 != 1) {
    //    mexErrMsgIdAndTxt("parallel:gpu:radixsort:InvalidInput","Input array must be M x 1.");
    //}
    //inArraySize = mrows1;

    /*
    if (ncols2 != 1) {
        mexErrMsgIdAndTxt("parallel:gpu:radixsort:InvalidInput","Input array must be M x 1.");
    }
    if (mrows1 != mrows2) {
        mexErrMsgIdAndTxt("parallel:gpu:radixsort:InvalidInput","Input arrays' size is different.");
    }
    */

    //plhs[0] = mxCreateDoubleMatrix((mwSize)inArraySize,(mwSize)2,mxREAL);
    //plhs[1] = mxCreateNumericArray(filter_dims, (mwSize* ) filter_size, mxSINGLE_CLASS, mxREAL);
    // return pointer to n-d numeric array
    plhs[0] = mxCreateNumericArray(image_dims, (mwSize* ) image_size, mxSINGLE_CLASS, mxREAL);
    //outArray = (float *) mxGetData(mxCreateNumericArray(image_dims, (mwSize* ) image_size, mxSINGLE_CLASS, mxREAL));
    // FIXME this probably doesn't work
    outArray = (float *) mxGetData(plhs[0]);

    // generate params
    int algo = 0; // forward convolve
    int benchmark = 0; //time it
    /* create a pointer to the real data in the input array,  */
    float *image = (float *) mxGetData(prhs[0]); // GetData returns type void *
    float *filter = (float *) mxGetData(prhs[1]); // GetData returns type void *
    //cufftutils::conv_handler(image, filter, outArray, algo, image_size, filter_size, benchmark);
    //cudnnutils::conv_handler(image, filter, outArray, algo, image_size, filter_size, benchmark);
}

