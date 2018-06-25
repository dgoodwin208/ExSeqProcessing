#ifndef __MX_ERROR_H__
#define __MX_ERROR_H__

#include <string>
#include "mex.h"
#include "matrix.h"

namespace mexutils {

void throwErrorMessage(const std::string& errMessage) {
    mxArray *arg = mxCreateString(errMessage.c_str());
    mexCallMATLAB(0, 0, 1, &arg, "error");
}

}

#endif // __MX_ERROR_H__

