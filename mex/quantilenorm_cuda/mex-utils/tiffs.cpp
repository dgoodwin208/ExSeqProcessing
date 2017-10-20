#include <string>
#include <vector>
#include <cstdint>

#include "mex.h"
#include "matrix.h"
#include "tiffs.h"

namespace mexutils {

void gettiffinfo(const std::string& filename, size_t& image_width, size_t& image_height, size_t& num_slices)
{
    mxArray *exception = NULL;

    mxArray *mx_imfinfo = NULL;
    mxArray *mx_filename = mxCreateString(filename.c_str());
    exception = mexCallMATLABWithTrap(1, &mx_imfinfo, 1, &mx_filename, "imfinfo");

    if (exception != NULL) {
        mxDestroyArray(mx_filename);
        mexCallMATLAB(0, (mxArray **)NULL, 1, &exception, "throw");
    }

    num_slices = mxGetM(mx_imfinfo);
    mexPrintf("slices=%d\n", num_slices);

    mxArray *mx_image_height = NULL;
    mxArray *mx_image_width  = NULL;
    if (mx_imfinfo != NULL) {
        mx_image_height = mxGetField(mx_imfinfo, 0, "Height");
        mx_image_width  = mxGetField(mx_imfinfo, 0, "Width");
        image_height = mxGetScalar(mx_image_height);
        image_width  = mxGetScalar(mx_image_width);
        mexPrintf("height=%d, width=%d\n", image_height, image_width);
    }

    mxDestroyArray(mx_imfinfo);
    mxDestroyArray(mx_filename);
}

void loadtiff(const std::string& filename, const size_t slice_start, const size_t slice_end, std::shared_ptr<std::vector<uint16_t>> image)
{
    mxArray *exception = NULL;

    for (int n_slice = slice_start; n_slice <= slice_end; n_slice++) {
        mxArray *data = NULL;
        mxArray *args[2] = { mxCreateString(filename.c_str()), mxCreateDoubleScalar(n_slice + 1) };
        exception = mexCallMATLABWithTrap(1, &data, 2, args, "imread");
        mxDestroyArray(args[0]);
        mxDestroyArray(args[1]);

        if (exception != NULL) {
            mexCallMATLAB(0, (mxArray **)NULL, 1, &exception, "throw");
        }

        uint16_t *values = (uint16_t *)mxGetData(data);
        size_t data_size = mxGetNumberOfElements(data);
//        mexPrintf("image size=%d,data_size=%d\n", image->size(), data_size);
//        mexPrintf("values[0]=%u, values[1000]=%u, values[end]=%u\n", values[0], values[1000], values[data_size-1]);
//        std::copy(values, values+data_size, image->begin());
        std::copy(values, values+data_size, std::back_inserter(*image));
        mxDestroyArray(data);
    }
}

}// namespace mexutils

