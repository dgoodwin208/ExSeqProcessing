TARGET=lib-cuda-utils lib-cudnn-utils lib-tests convn_cuda.mexa64

SOURCES := $(wildcard convn_cuda.cpp)
OBJS := $(SOURCES:.cpp=.o)

MATLAB_GPU_INC=/usr/local/MATLAB/R2017a/toolbox/distcomp/gpu/extern/include
CUDA_INC=/usr/local/cuda/include
CUDA_LIB=/usr/local/cuda/lib64
CUDNN_DIR=cudnn-utils

all: $(TARGET)

convn_cuda.mexa64: $(OBJS) $(CUDNN_DIR)/libcudnnutils.a
	mex -compatibleArrayDims -outdir ../ -output $@ $(OBJS) $(CUDNN_DIR)/libcudnnutils.a -lmwgpu -L$(CUDA_LIB) -lcudart -ltbb -lcudnn

.cpp.o: $(SOURCES)
	mex -compatibleArrayDims -c $< -I../include -I$(MATLAB_GPU_INC) -I$(CUDA_INC) -I$(CUDNN_DIR)

lib-cuda-utils:
	make -C cuda-utils

lib-cudnn-utils:
	make -C cudnn-utils libcudnnutils.a

lib-tests:
	make -C tests

clean:
	-rm *.mexa64 *.o
	@make -C tests clean
	@make -C cuda-utils clean
	@make -C cudnn-utils clean

