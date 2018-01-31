#include <arrayfire.h>
#include <mex.h>


void testBackend()
{
    af::info();
    af_print(af::randu(5,4));
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{

    /* Check for proper number of input and output arguments */    
    if (nrhs != 2) {
        mexErrMsgIdAndTxt( "convn_cuda:InvalidInput","Two input arguments required.");
    } 
    if (nlhs > 1){
        mexErrMsgIdAndTxt( "convn_cuda:InvalidOutput","Only 0 or 1 output arguments.");
    }

    /* make sure input array arguments are type double */
    if ( !mxIsSingle(prhs[0]) || 
          mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("convn_cuda:InvalidInput","Input array must be type real single valued.");
    }
    if ( !mxIsSingle(prhs[1]) || 
          mxIsComplex(prhs[1])) {
        mexErrMsgIdAndTxt("convn_cuda:InvalidInput","Input array must be type real single valued.");
    }

    // Handle image and filter array sizes
    int *image_size;
    int image_dims = (int) mxGetNumberOfDimensions(prhs[0]);
    image_size = (int *) mxGetDimensions(prhs[0]);
    /* create a pointer to the real data in the input array,  */
    float* image = (float*) mxGetData(prhs[0]); // GetData returns type void *
    af::array image_array(image_size[0], image_size[1], image_size[2], image, afHost);
    printf("image size: %d, %d, %d\n", image_size[0], image_size[1], image_size[2]);

    //const size_t *filter_size;
    int *filter_size;
    //const mwSize filter_dims = mxGetNumberOfDimensions(prhs[1]);
    int filter_dims = (int) mxGetNumberOfDimensions(prhs[1]);
    filter_size = (int *) mxGetDimensions(prhs[1]);
    float* filter = (float*) mxGetData(prhs[1]); // GetData returns type void *
    af::array filter_array(filter_size[0], filter_size[1], filter_size[1], filter, afHost);
    printf("filter size: %d, %d, %d\n", filter_size[0], filter_size[1], filter_size[2]);

    plhs[0] = mxCreateNumericArray(image_dims, (mwSize* ) image_size, mxSINGLE_CLASS, mxREAL);
    float* outArray = (float*) mxGetData(plhs[0]);

    try {
        printf("Backend count: %d", af::getBackendCount());
        printf("Trying CUDA Backend\n");
        af::setBackend(AF_BACKEND_CUDA);
        testBackend();
    } catch (af::exception& e) {
        printf("Caught exception when trying CUDA backend");
        fprintf(stderr, "%s\n", e.what());
    }

    af::array output = af::convolve(image_array, filter_array);
    output.host((void*)outArray);
}
