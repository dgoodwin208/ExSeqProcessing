/*=================================================================
 * averageSameRanks.c - average values in the same rank, and
 *                      relace it
 *
 *  outArray = averageSameRanks(inArray, inSameRankIdxList)
 *
 *  inArray:  M array
 *  inSameRankIdxList: N array
 *  outArray: M array
 *
 *=================================================================*/
 

#include <stdio.h>
#include <string.h>
#include "mex.h"

void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{ 
    double *inArray;                /* M input array */
    double *inSameRankIdxList;      /* M input array */
    size_t mrows, ncols;
    size_t inArraySize;             /* size of array */
    size_t inSameRankIdxListSize;   /* size of sameRankIdxList */
    double *outArray;               /* M output array */
    int i,j,k;
    int s_idx,e_idx;
    double d_mean;
    double count;

    /* Check for proper number of input and output arguments */    
    if (nrhs != 2) {
        mexErrMsgIdAndTxt( "MATLAB:averageSameRanks:minrhs","Two input arguments required.");
    } 
    if (nlhs > 1){
        mexErrMsgIdAndTxt( "MATLAB:averageSameRanks:maxrhs","Too many output arguments.");
    }

    /* make sure input array arguments are type double */
    if ( !mxIsDouble(prhs[0]) || 
          mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MATLAB:averageSameRanks:notDouble","Input array must be type double.");
    }
    if ( !mxIsNumeric(prhs[1])) {
        mexErrMsgIdAndTxt("MATLAB:averageSameRanks:notNumeric","Input array must be type numeric.");
    }


    /* create a pointer to the real data in the input array  */
    inArray = mxGetPr(prhs[0]);
    inSameRankIdxList = mxGetPr(prhs[1]);

    /* get dimensions of the input array */
    mrows = mxGetM(prhs[0]);
    ncols = mxGetN(prhs[0]);
    if (ncols != 1) {
        mexErrMsgIdAndTxt("MATLAB:averageSameRanks:invalidInputSize","Input array must be M x 1.");
    }
    inArraySize = mrows;

    mrows = mxGetM(prhs[1]);
    ncols = mxGetN(prhs[1]);
    if (ncols != 1) {
        mexErrMsgIdAndTxt("MATLAB:averageSameRanks:invalidInputSize","Input array must be M x 1.");
    }
    inSameRankIdxListSize = mrows;
    //mexPrintf("inArray#=%d,sameRankIdxList#=%d\n",inArraySize,inSameRankIdxListSize);

    /* create the output array */
    plhs[0] = mxCreateDoubleMatrix((mwSize)inArraySize,(mwSize)1,mxREAL);

    /* get a pointer to the real data in the output array */
    outArray = mxGetPr(plhs[0]);
    memcpy((void *)outArray, (void *)inArray, sizeof(double)*inArraySize); 

    s_idx = 0;
    e_idx = 0;
    while (s_idx < inSameRankIdxListSize) {
        //mexPrintf("s_idx=%d,e_idx=%d\n",s_idx,e_idx);
        if ((e_idx+1 < inSameRankIdxListSize) &&
            ((int)inSameRankIdxList[e_idx]+1 == (int)inSameRankIdxList[e_idx+1])) {
            e_idx++;
            continue;
        }

        //mexPrintf("average\n");
        count = 0;
        d_mean = 0.0;
        for (i = (int)inSameRankIdxList[s_idx]-1; i <= inSameRankIdxList[e_idx]; i++) {
            //mexPrintf("i=%d,v=%f\n",i,inArray[i]);
            d_mean += inArray[i];
            count++;
        }
        d_mean = d_mean / (double)count;
        //mexPrintf("d_mean=%f\n",d_mean);

        //mexPrintf("copy\n");
        for (i = (int)inSameRankIdxList[s_idx]-1; i <= inSameRankIdxList[e_idx]; i++) {
            //mexPrintf("i=%d,v=%f\n",i,inArray[i]);
            outArray[i] = d_mean;
        }

        s_idx = e_idx+1;
        e_idx = s_idx;
    }
}

