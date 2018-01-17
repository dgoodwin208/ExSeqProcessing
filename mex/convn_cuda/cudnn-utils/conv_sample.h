#ifndef __CUDNNUTILS_H__
#define __CUDNNUTILS_H__


namespace cudnnutils {

int conv_handler(double* hostI, double* hostF, double* hostO, int algo, int* dimA, int* filterdimA, int benchmark);

}

#endif
