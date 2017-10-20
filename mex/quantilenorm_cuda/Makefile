TARGET=lib-cuda-utils lib-mex-utils lib-utils quantilenorm_cuda.mexa64

#SOURCES := $(wildcard *.cpp)
SOURCES := $(wildcard quantilenorm_*.cpp)
OBJS := $(SOURCES:.cpp=.o)

MATLAB_GPU_INC=/usr/local/MATLAB/R2017a/toolbox/distcomp/gpu/extern/include
CUDA_INC=/usr/local/cuda/include
CUDA_LIB=/usr/local/cuda/lib64

all: $(TARGET)

quantilenorm_cuda.mexa64: $(OBJS) cuda-utils/libcudautils.a mex-utils/libmexutils.a utils/libutils.a
	mex -output $@ $(OBJS) -L./cuda-utils -lcudautils -L./mex-utils -lmexutils -L./utils -lutils -lmwgpu -L$(CUDA_LIB) -lcudart -ltbb
	@cp -a quantilenorm_cuda.mexa64 ../

.cpp.o: $(SOURCES)
	mex -c $< -I../include -I$(MATLAB_GPU_INC) -I$(CUDA_INC)

lib-cuda-utils:
	make -C cuda-utils

lib-mex-utils:
	make -C mex-utils

lib-utils:
	make -C utils

clean:
	-rm *.mexa64 *.o
	@make -C cuda-utils clean
	@make -C mex-utils clean
	@make -C utils clean

