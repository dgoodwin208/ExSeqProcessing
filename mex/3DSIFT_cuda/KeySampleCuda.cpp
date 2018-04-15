/* ==========================================================================
 * KeySampleCuda.cpp
 *
 * takes a (MxN) structure matrix and returns a new structure (1x1)
 * containing corresponding fields: for string input, it will be (MxN)
 * cell array; and for numeric (noncomplex, scalar) input, it will be (MxN)
 * vector of numbers with the same classID as input, such as int, double
 * etc..
 *
 *==========================================================================*/

#include "mex.h"
#include "cudautils.h"

/*  the gateway routine.  */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    const char **fnames;       /* pointers to field names */
    const mwSize *dims;
    mxArray    *tmp, *fout;
    char       *pdata=NULL;
    int        ifield, nfields;
    mxClassID  *classIDflags;
    mwIndex    jstruct;
    mwSize     NStructElems;
    mwSize     ndim;
    
    /* check proper input and output */
    if(nrhs!=4)
        mexErrMsgIdAndTxt( "MATLAB:KeySampleCuda:invalidNumInputs",
                "Four inputs required.");
    else if(nlhs != 2)
        mexErrMsgIdAndTxt( "MATLAB:KeySampleCuda:maxlhs",
                "2 output arguments required.");
    else if(!mxIsStruct(prhs[0]))
        mexErrMsgIdAndTxt( "MATLAB:KeySampleCuda:inputNotStruct",
                "Input 1 must be a structure.");
    else if(!mxIsDouble(prhs[1]))
        mexErrMsgIdAndTxt( "MATLAB:KeySampleCuda:inputNotStruct",
                "Input 1 must be type double.");
    else if(!mxIsStruct(prhs[2]))
        mexErrMsgIdAndTxt( "MATLAB:KeySampleCuda:inputNotStruct",
                "Input 3 must be a structure.");
    else if(!mxIsStruct(prhs[3]))
        mexErrMsgIdAndTxt( "MATLAB:KeySampleCuda:inputNotStruct",
                "Input 4 must be a structure.");
    int IndexSize = (int) mxGetData(mxGetField(prhs[2], 0, "IndexSize"));
    int nFaces = (int) mxGetData(mxGetField(prhs[2], 0, "nFaces"));
    mwSize* dims = {IndexSize, IndexSize, IndexSize, nFaces};
    // FIXME this must be initialized to zero
    plhs[0] = mxCreateNumericArray(4, dims, mxDouble_CLASS, mxREAL);
    double* index = plhs[0];
    plhs[1] = mxCreateStructMatrix(1,1,1,"test"); // precomp_grads struct
    cudautils::SIFT_handler();
}
