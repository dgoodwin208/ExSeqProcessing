#include <string>
#include <vector>
#include <cstdint>

#include "mex.h"
#include "matrix.h"
#include "hdf5.h"
#include "mx_error.h"

namespace mexutils {

void gethdf5finfo(const std::string& filename, size_t& image_width, size_t& image_height, size_t& num_slices)
{
    mxArray *exception = NULL;

    mxArray *mx_h5info = NULL;
    mxArray *mx_h5info_args[2] = { mxCreateString(filename.c_str()), mxCreateString("/image") };
    exception = mexCallMATLABWithTrap(1, &mx_h5info, 2, mx_h5info_args, "h5info");

    mxDestroyArray(mx_h5info_args[0]);
    mxDestroyArray(mx_h5info_args[1]);
    if (exception != NULL) {
        mexCallMATLAB(0, (mxArray **)NULL, 1, &exception, "throw");
    }

    if (mx_h5info == NULL) {
        std::string errMessage("hdf5 info not exist");
        throwErrorMessage(errMessage);
    }

    mxArray *mx_dataspace = mxGetField(mx_h5info, 0, "Dataspace");
    if (mx_dataspace == NULL) {
        std::string errMessage("hdf5 Dataspace not exist");
        throwErrorMessage(errMessage);
    }

    mxArray *mx_size = mxGetField(mx_dataspace , 0, "Size");
    if (mx_size == NULL) {
        std::string errMessage("hdf5 Dataspace.Size not exist");
        throwErrorMessage(errMessage);
    }
    if (! mxIsDouble(mx_size)) {
        std::string errMessage("hdf5 Dataspace.Size is not double-type");
        throwErrorMessage(errMessage);
    }
    if (mxGetNumberOfElements(mx_size) != 3) {
        std::string errMessage("hdf5 Dataspace.Size does not have 3 elements");
        throwErrorMessage(errMessage);
    }

    double *dim_size = mxGetPr(mx_size);

    image_height = dim_size[0];
    image_width  = dim_size[1];
    num_slices   = dim_size[2];
    mexPrintf("height=%d, width=%d, num_slices=%d\n", image_height, image_width, num_slices);

    mxDestroyArray(mx_h5info);
}

void loadhdf5(const std::string& filename, const size_t slice_start, const size_t slice_end, const size_t image_height, const size_t image_width, std::shared_ptr<std::vector<uint16_t>> image)
{
    mxArray *exception = NULL;

    mxArray *data = NULL;
    mxArray *args[4] = {
        mxCreateString(filename.c_str()),
        mxCreateString("/image"),
        mxCreateDoubleMatrix((mwSize)1, (mwSize)3, mxREAL),
        mxCreateDoubleMatrix((mwSize)1, (mwSize)3, mxREAL)
    };

    double *mat_start = mxGetPr(args[2]);
    double *mat_count = mxGetPr(args[3]);
    mat_start[0] = 1;
    mat_start[1] = 1;
    mat_start[2] = slice_start + 1;
    mat_count[0] = image_height;
    mat_count[1] = image_width;
    mat_count[2] = slice_end - slice_start + 1;

    exception = mexCallMATLABWithTrap(1, &data, 4, args, "h5read");
    mxDestroyArray(args[0]);
    mxDestroyArray(args[1]);
    mxDestroyArray(args[2]);
    mxDestroyArray(args[3]);

    if (exception != NULL) {
        mexCallMATLAB(0, (mxArray **)NULL, 1, &exception, "throw");
    }

    uint16_t *values = (uint16_t *)mxGetData(data);
    size_t data_size = mxGetNumberOfElements(data);
    std::copy(values, values+data_size, std::back_inserter(*image));
    mxDestroyArray(data);
}

}// namespace mexutils

