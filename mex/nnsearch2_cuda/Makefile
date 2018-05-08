.SUFFIXES: .cu

TARGET=nnsearch2_cuda.mexa64 nnsearch2_cuda

SRCS := $(wildcard nnsearch2_*.cpp)
OBJS := $(SRCS:.cpp=.o)
CU_SRCS := $(wildcard *.cu)
CU_OBJS := $(CU_SRCS:.cu=.o)

MATLAB_GPU_INC=/usr/local/MATLAB/R2018a/toolbox/distcomp/gpu/extern/include
CUDA_INC=/usr/local/cuda/include
CUDA_LIB=/usr/local/cuda/lib64

SPDLOG_DIR=../include
CUDA_UTILS_DIR=../cuda-utils

CPPFLAGS+=-I$(CUDA_UTILS_DIR) -I$(SPDLOG_DIR)
CXXFLAGS+=-std=c++11 -Wall -Wextra -pthread
NVCCFLAGS=-std=c++11 --gpu-architecture=sm_61

all: $(TARGET)

nnsearch2_cuda.mexa64: $(OBJS) $(CU_OBJS) $(CUDA_UTILS_DIR)/libcudautils.a
	mex -output $@ $^ -lmwgpu -L$(CUDA_LIB) -lcudart
	@cp -a $@ ../

.cpp.o: $(SRCS)
	mex -c $< $(CPPFLAGS) -I$(MATLAB_GPU_INC) -I$(CUDA_INC)

#lib-cuda-utils:
$(CUDA_UTILS_DIR)/libcudautils.a: $(CUDA_UTILS_DIR)/*.cu $(CUDA_UTILS_DIR)/*.h
	make -C $(CUDA_UTILS_DIR)

.cu.o: $(CU_SRCS) *.h
	nvcc $(NVCCFLAGS) $(CPPFLAGS) -Xcompiler -fPIC -c -o $@ $<

nnsearch2_cuda: $(CU_OBJS) $(CUDA_UTILS_DIR)/libcudautils.a
	g++ -o $@ $^ -L$(CUDA_LIB) -lcudart -lpthread

clean:
	-rm -f $(TARGET) *.o

