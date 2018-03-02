#ifndef __CUFFTUTILS_H__
#define __CUFFTUTILS_H__

namespace cufftutils {

    int conv_handler(float* hostI, float* hostF, float* hostO, int algo, 
            int* dimA, int* filterdimA, int benchmark);

}

#endif
