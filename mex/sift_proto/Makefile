.SUFFIXES: .cu

TARGET=sift_cuda.mexa64 sift_cuda

SRCS := $(wildcard sift_*.cpp)
OBJS := $(SRCS:.cpp=.o)
CU_SRCS := $(wildcard *.cu)
CU_OBJS := $(CU_SRCS:.cu=.o)
HEADERS := $(wildcard *.h)

MATLAB_GPU_INC=/usr/local/MATLAB/R2018a/toolbox/distcomp/gpu/extern/include
CUDA_INC=/usr/local/cuda/include
CUDA_LIB=/usr/local/cuda/lib64

SPDLOG_DIR=../include
CUDA_UTILS_DIR=../cuda-utils

CPPFLAGS+=-I$(CUDA_UTILS_DIR) -I$(SPDLOG_DIR)
CXXFLAGS+=-std=c++11 -Wall -Wextra -pthread
NVCCFLAGS=-std=c++11 --gpu-architecture=sm_61

all: $(TARGET)

sift_cuda.mexa64: $(OBJS) $(CU_OBJS) $(CUDA_UTILS_DIR)/libcudautils.a
	mex -compatibleArrayDims -g -output $@ $^ -lmwgpu -L$(CUDA_LIB) -lcudart -lpthread
	@cp -a $@ ../

%.o: %.cpp $(SRCS)
	mex -compatibleArrayDims -g -c $< $(CPPFLAGS) -I$(MATLAB_GPU_INC) -I$(CUDA_INC)

#lib-cuda-utils:
$(CUDA_UTILS_DIR)/libcudautils.a: $(CUDA_UTILS_DIR)/*.cu $(CUDA_UTILS_DIR)/*.h
	make -C $(CUDA_UTILS_DIR)

%.o: %.cu $(CU_SRCS)
	nvcc $(NVCCFLAGS) $(CPPFLAGS) -Xcompiler -fPIC -g -G -c -o $@ $<

sift_cuda: $(CU_OBJS) $(CUDA_UTILS_DIR)/libcudautils.a
	g++ -g -o $@ $^ -L$(CUDA_LIB) -lcudart -lpthread

clean:
	-rm -f $(TARGET) *.o
	-rm -f ../sift_cuda.mexa64

