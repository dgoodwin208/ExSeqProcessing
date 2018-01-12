TARGET=lib-cuda-utils lib-tests # convn_cuda.mexa64

#SOURCES := $(wildcard *.cpp)
SOURCES := $(wildcard convn_*.cpp)
OBJS := $(SOURCES:.cpp=.o)

MATLAB_GPU_INC=/usr/local/MATLAB/R2017a/toolbox/distcomp/gpu/extern/include
CUDA_INC=/usr/local/cuda/include
CUDA_LIB=/usr/local/cuda/lib64

all: $(TARGET)

#convn_cuda.mexa64: $(OBJS) cuda-utils/libcudautils.a 
	#mex -output $@ $(OBJS) -L./cuda-utils -lcudautils -lmexutils -lutils -lmwgpu -L$(CUDA_LIB) -lcudart -ltbb
	#@cp -a convn_cuda.mexa64 ../

#.cpp.o: $(SOURCES)
	#mex -c $< -I../include -I$(MATLAB_GPU_INC) -I$(CUDA_INC)

lib-cuda-utils:
	make -C cuda-utils

lib-tests:
	make -C tests

lib-cudnn-utils:
	make -C cudnn-utils

clean:
	-rm *.mexa64 *.o
	@make -C tests clean
	@make -C cuda-utils clean
	@make -C cudnn-utils clean

