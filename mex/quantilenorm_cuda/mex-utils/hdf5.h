#ifndef __HDF5_H__
#define __HDF5_H__

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace mexutils {

void gethdf5finfo(const std::string& filename, size_t& image_width, size_t& image_height, size_t& num_slices);

void loadhdf5(const std::string& filename, const size_t slice_start, const size_t slice_end, const size_t image_height, const size_t image_width, std::shared_ptr<std::vector<uint16_t>> image);

}

#endif // __HDF5_H__


