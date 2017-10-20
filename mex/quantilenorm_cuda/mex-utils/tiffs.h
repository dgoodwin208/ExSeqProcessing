#ifndef __TIFFS_H__
#define __TIFFS_H__

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace mexutils {

void gettiffinfo(const std::string& filename, size_t& image_width, size_t& image_height, size_t& num_slices);

void loadtiff(const std::string& filename, const size_t slice_start, const size_t slice_end, std::shared_ptr<std::vector<uint16_t>> image);

}

#endif // __TIFFS_H__

