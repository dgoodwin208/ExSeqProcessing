/*=================================================================
 * mergearrays.c - merge 2 sorted arrays to one array
 *                the first column is sorted
 *
 *  outMatrix = mergearrays(inMatrix[0], inMatrix[1])
 *
 *  inMatrix:  MxN matrix
 *  outMatrix: (2*M)xN matrix
 *
 *=================================================================*/
 

#include <stdio.h>
#include <string.h>
#include <limits.h>
#include "mex.h"


void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{ 
    double *inMatrix[2];            /* 4xNx2 input matrix */
    size_t mrows[2];                /* row size of matrix */
    size_t ncols[2];                /* col size of matrix */
    double *outMatrix;              /* output matrix */
    int i,j,k;
    int idx[2];
    double min_val;
    size_t total_mrows;

    /* Check for proper number of input and output arguments */    
    if (nrhs != 2) {
        mexErrMsgIdAndTxt( "MATLAB:mergearrays:minrhs","Two input arguments required.");
    }
    if (nlhs > 1){
        mexErrMsgIdAndTxt( "MATLAB:mergearrays:maxrhs","Too many output arguments.");
    }

    /* make sure input matrix arguments are type double */
    for (i = 0; i < 2; i++) {
        if ( !mxIsDouble(prhs[i]) || 
              mxIsComplex(prhs[i])) {
            mexErrMsgIdAndTxt("MATLAB:mergearrays:notDouble","Input matrix must be type double.");
        }
    }


    /* create a pointer to the real data in the input matrix  */
    for (i = 0; i < 2; i++) {
        inMatrix[i] = mxGetPr(prhs[i]);
    }

    /* get dimensions of the input matrix */
    for (i = 0; i < 2; i++) {
        mrows[i] = mxGetM(prhs[i]);
        ncols[i] = mxGetN(prhs[i]);
    }
    if (ncols[0] != 2 || ncols[0] != ncols[1]) {
        mexErrMsgIdAndTxt("MATLAB:mergearrays:invalidInputSize","Input matrix columns must be 2 and the same column size.");
    }

    /* create the output matrix */
    total_mrows = mrows[0]+mrows[1];
    plhs[0] = mxCreateDoubleMatrix((mwSize)total_mrows,(mwSize)ncols[0],mxREAL);

    /* get a pointer to the real data in the output matrix */
    outMatrix = mxGetPr(plhs[0]);

    for (i = 0; i < 2; i++) {
        idx[i] = 0;
    }
    k = 0;
    while ((idx[0] < mrows[0]) || (idx[1] < mrows[1])) {
        j = 0;
        min_val = DBL_MAX;
        for (i = 0; i < 2; i++) {
            if (idx[i] < mrows[i]) {
                if (inMatrix[i][idx[i]] < min_val) {
                    min_val = inMatrix[i][idx[i]];
                    j = i;
                }
            }
        }

        outMatrix[k            ] = inMatrix[j][idx[j]         ];
        outMatrix[k+total_mrows] = inMatrix[j][idx[j]+mrows[j]];
        idx[j]++;
        k++;
    }

    return;
}

