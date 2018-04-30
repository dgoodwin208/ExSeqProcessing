.SUFFIXES: .cu

TARGET=nearestinterp_cuda.mexa64 nearestinterp_cuda

SRCS := $(wildcard nearestinterp_*.cpp)
OBJS := $(SRCS:.cpp=.o)
CU_SRCS := $(wildcard *.cu)
CU_OBJS := $(CU_SRCS:.cu=.o)

MATLAB_GPU_INC=/usr/local/MATLAB/R2018a/toolbox/distcomp/gpu/extern/include
CUDA_INC=/usr/local/cuda/include
CUDA_LIB=/usr/local/cuda/lib64

NVCC_FLAGS=--gpu-architecture=sm_61

SPDLOG_DIR=../include
CUDA_UTILS_DIR=../cuda-utils

all: $(TARGET)

nearestinterp_cuda.mexa64: $(OBJS) $(CU_OBJS) $(CUDA_UTILS_DIR)/libcudautils.a
	mex -output $@ $^ -lmwgpu -L$(CUDA_LIB) -lcudart -lpthread
	@cp -a $@ ../

.cpp.o: $(SRCS)
	mex -c $< -I$(SPDLOG_DIR) -I$(MATLAB_GPU_INC) -I$(CUDA_INC) -I$(CUDA_UTILS_DIR)

#lib-cuda-utils:
$(CUDA_UTILS_DIR)/libcudautils.a: $(CUDA_UTILS_DIR)/*.cu $(CUDA_UTILS_DIR)/*.h
	make -C $(CUDA_UTILS_DIR)

.cu.o: $(CU_SRCS) *.h
	nvcc -std=c++11 $(NVCC_FLAGS) -Xcompiler -fPIC -c -o $@ $< -I$(SPDLOG_DIR) -I$(CUDA_INC) -I$(CUDA_UTILS_DIR)

nearestinterp_cuda: $(CU_OBJS) $(CUDA_UTILS_DIR)/libcudautils.a
	g++ -o $@ $^ -L$(CUDA_LIB) -lcudart -lpthread

clean:
	-rm -f $(TARGET) *.o

