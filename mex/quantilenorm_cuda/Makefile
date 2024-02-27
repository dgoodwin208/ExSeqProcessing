.SUFFIXES: .cu

TARGET=quantilenorm_cuda.mexa64

SRCS := $(wildcard quantilenorm_*.cpp)
OBJS := $(SRCS:.cpp=.o)
CU_SRCS := $(wildcard *.cu)
CU_OBJS := $(CU_SRCS:.cu=.o)

MATLAB_GPU_INC=/usr/local/MATLAB/R2020b/toolbox/distcomp/gpu/extern/include
CUDA_INC=/usr/local/cuda/include
CUDA_LIB=/usr/local/cuda/lib64

SPDLOG_DIR=../include
CUDA_UTILS_DIR=../cuda-utils
MEX_UTILS_DIR=mex-utils
UTILS_DIR=utils
GLOBAL_UTILS_DIR=../utils

CPPFLAGS+=-DNDEBUG -I$(CUDA_UTILS_DIR) -I$(GLOBAL_UTILS_DIR) -I$(SPDLOG_DIR)
CXXFLAGS+=-std=c++11 -Wall -Wextra -pthread
NVCCFLAGS=-std=c++11 --gpu-architecture=sm_61

all: $(TARGET)

quantilenorm_cuda.mexa64: $(OBJS) $(CU_OBJS) $(CUDA_UTILS_DIR)/libcudautils.a $(MEX_UTILS_DIR)/libmexutils.a $(UTILS_DIR)/libutils.a
	mex -output $@ $^ -lmwgpu -L$(CUDA_LIB) -lcudart -lpthread -ltbb
	@cp -a quantilenorm_cuda.mexa64 ../bin/

%.o: %.cpp $(SRCS)
	mex -c $< $(CPPFLAGS) -I$(MATLAB_GPU_INC) -I$(CUDA_INC)

$(CUDA_UTILS_DIR)/libcudautils.a: $(CUDA_UTILS_DIR)/*.cu $(CUDA_UTILS_DIR)/*.h
	make -C $(CUDA_UTILS_DIR)

%.o: %.cu $(CU_SRCS)
	nvcc $(NVCCFLAGS) $(CPPFLAGS) -Xcompiler -fPIC -c -o $@ $<

$(MEX_UTILS_DIR)/libmexutils.a: $(MEX_UTILS_DIR)/*.cpp $(MEX_UTILS_DIR)/*.h
	make -C $(MEX_UTILS_DIR)

$(UTILS_DIR)/libutils.a: $(UTILS_DIR)/*.cpp $(UTILS_DIR)/*.h
	make -C $(UTILS_DIR)

clean:
	rm -f *.mexa64 *.o
	@make -C $(MEX_UTILS_DIR) clean
	@make -C $(UTILS_DIR) clean

