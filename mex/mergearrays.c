/*=================================================================
 * mergearrays.c - merge two sorted arrays to one array
 *                 the first column is sorted
 *
 *  outMatrix = mergearrays(inMatrix[0], inMatrix[1])
 *
 *  inMatrix:  MxN matrix
 *  outMatrix: (2*M)xN matrix
 *
 *=================================================================*/
 

#include <stdio.h>
#include <string.h>
#include "mex.h"


void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{ 
    double *inMatrix[2];            /* 2xNx2 input matrix */
    size_t mrows[2];                /* row size of matrix */
    size_t ncols[2];                /* col size of matrix */
    double *outMatrix;              /* output matrix */
    mwSize i,j,k;

    /* Check for proper number of input and output arguments */    
    if (nrhs != 2) {
        mexErrMsgIdAndTxt( "MATLAB:mergearrays:minrhs","Two or three input arguments required.");
    } 
    if (nlhs > 1){
        mexErrMsgIdAndTxt( "MATLAB:mergearrays:maxrhs","Too many output arguments.");
    }

    /* make sure the first and second input arguments are type double */
    if ( !mxIsDouble(prhs[0]) || 
          mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MATLAB:mergearrays:notDouble","Input matrix must be type double.");
    }
    if ( !mxIsDouble(prhs[1]) || 
          mxIsComplex(prhs[1])) {
        mexErrMsgIdAndTxt("MATLAB:mergearrays:notDouble","Input matrix must be type double.");
    }


    /* create a pointer to the real data in the input matrix  */
    inMatrix[0] = mxGetPr(prhs[0]);
    inMatrix[1] = mxGetPr(prhs[1]);

    /* get dimensions of the input matrix */
    mrows[0] = mxGetM(prhs[0]);
    ncols[0] = mxGetN(prhs[0]);

    mrows[1] = mxGetM(prhs[1]);
    ncols[1] = mxGetN(prhs[1]);
    if (ncols[0] != 2 || ncols[0] != ncols[1]) {
        mexErrMsgIdAndTxt("MATLAB:mergearrays:invalidInputSize","Input matrix must be the same column size.");
    }

    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix((mwSize)(mrows[0]+mrows[1]),(mwSize)ncols[0],mxREAL);

    /* get a pointer to the real data in the output matrix */
    outMatrix = mxGetPr(plhs[0]);

    i = 0; j = 0; k = 0;
    while (i < mrows[0] && j < mrows[1]) {
        if (inMatrix[0][i] <= inMatrix[1][j]) {
            outMatrix[k                  ] = inMatrix[0][i         ];
            outMatrix[k+mrows[0]+mrows[1]] = inMatrix[0][i+mrows[0]];
            i++;
        } else {
            outMatrix[k                  ] = inMatrix[1][j         ];
            outMatrix[k+mrows[0]+mrows[1]] = inMatrix[1][j+mrows[1]];
            j++;
        }
        k++;
    }

    for (; i < mrows[0]; i++) {
        outMatrix[k                  ] = inMatrix[0][i         ];
        outMatrix[k+mrows[0]+mrows[1]] = inMatrix[0][i+mrows[0]];
        k++;
    }
    for (; j < mrows[1]; j++) {
        outMatrix[k                  ] = inMatrix[1][j         ];
        outMatrix[k+mrows[0]+mrows[1]] = inMatrix[1][j+mrows[1]];
        k++;
    }

    return;
}

