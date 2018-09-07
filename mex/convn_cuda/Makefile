TARGET=lib-cufft-utils lib-utils convn_cuda.o convn_cuda.mexa64 cmd_convn_cuda  lib-tests # convn_1GPU_cuda.o fft3_cuda.o convn_1GPU_cuda.mexa64 fft3_cuda.mexa64 lib-cuda-utils lib-cudnn-utils convn_arrayfire.mexa64 convn_arrayfire_cuda.mexa64 

SOURCES := $(wildcard convn_cuda.cpp fft3_cuda.cpp)
OBJS := $(SOURCES:.cpp=.o)

MATLAB_GPU_INC=/usr/local/MATLAB/R2017a/toolbox/distcomp/gpu/extern/include
CUDA_INC=/usr/local/cuda/include
CUDA_LIB=/usr/local/cuda/lib64
CUDNN_DIR=cudnn-utils
CUFFT_DIR=cufft-utils

SPDLOG_DIR=../include
CUDA_UTILS_DIR=../cuda-utils
TESTS_DIR=tests/

CPPFLAGS+=-I$(CUDA_UTILS_DIR) -I$(SPDLOG_DIR)
CXXFLAGS+=-std=c++11 -Wall -Wextra -pthread
NVCCFLAGS=-std=c++11 --gpu-architecture=sm_61

all: $(TARGET)

convn_cuda.mexa64: convn_cuda.o $(CUFFT_DIR)/libcufftutils.a
	mex -compatibleArrayDims -outdir ../ -output $@ $^ $(CUFFT_DIR)/libcufftutils.a -lmwgpu -L$(CUDA_LIB) -I$(CUDA_INC) -lcufft -lcudart

#convn_1GPU_cuda.mexa64: convn_1GPU_cuda.o $(CUFFT_DIR)/libcufftutils.a
	#mex -compatibleArrayDims -outdir ../ -output $@ $^ $(CUFFT_DIR)/libcufftutils.a -lmwgpu -L$(CUDA_LIB) -I$(CUDA_INC) -lcufft -lcudart

convn_cuda.o: convn_cuda.cpp
	mex -compatibleArrayDims -c $^ -I../include -I$(MATLAB_GPU_INC) -I$(CUDA_INC) -I$(CUFFT_DIR)

#convn_1GPU_cuda.o: convn_1GPU_cuda.cpp
	#mex -compatibleArrayDims -c $^ -I../include -I$(MATLAB_GPU_INC) -I$(CUDA_INC) -I$(CUFFT_DIR)

cmd_convn_cuda.o: cmd_convn_cuda.cu
	#nvcc $(NVCCFLAGS) $(CPPFLAGS) -Xcompiler -fPIC -O2 -c -o $@ $<
	nvcc $(NVCCFLAGS) $(CPPFLAGS) -Xcompiler -fPIC -g -G -O0 -c -o $@ $<

cmd_convn_cuda: cmd_convn_cuda.o $(CUFFT_DIR)/libcufftutils.a
	#g++ -O2 -o $@ $^ -L$(CUDA_LIB) -lcudart -lpthread
	g++ -fvar-tracking-assignments -g -O0 -o $@ $^ -L$(CUDA_LIB) -lcudart -lpthread -lcufft

lib-cufft-utils:
	make -C cufft-utils

lib-utils:
	make -C utils

#.cpp.o: $(SOURCES)
	#mex -compatibleArrayDims -c $< -I../include -I$(MATLAB_GPU_INC) -I$(CUDA_INC) -I$(CUFFT_DIR)
	##mex -compatibleArrayDims -c $< -I../include -I$(MATLAB_GPU_INC) -I$(CUDA_INC) -I$(CUDNN_DIR)

# convn_cuda.mexa64: $(OBJS) $(CUDNN_DIR)/libcudnnutils.a
#	mex -compatibleArrayDims -outdir ../ -output $@ $(OBJS) $(CUDNN_DIR)/libcudnnutils.a -lmwgpu -L$(CUDA_LIB) -lcudart -lcudnn #-ltbb 

#lib-cuda-utils:
#	make -C cuda-utils

#lib-cudnn-utils:
#	make -C cudnn-utils libcudnnutils.a

lib-tests:
	make -C tests

clean:
	-rm *.mexa64 *.o
	-rm ../convn_cuda.mexa64
	@make -C tests clean
	@make -C utils clean
	@make -C cufft-utils clean
	#-rm ../convn_1GPU_cuda.mexa64
	#@make -C cuda-utils clean
	#@make -C cudnn-utils clean

