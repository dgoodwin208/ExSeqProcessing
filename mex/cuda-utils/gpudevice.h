#ifndef __GPUDEVICE_H__
#define __GPUDEVICE_H__

namespace cudautils {

int get_gpu_num();

void get_gpu_mem_size(size_t& free_size, size_t& total_size);

void resetDevice();

}

#endif // __GPUDEVICE_H__

