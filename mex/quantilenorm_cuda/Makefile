TARGET=quantilenorm_cuda.mexa64

SRCS := $(wildcard quantilenorm_*.cpp)
OBJS := $(SRCS:.cpp=.o)

MATLAB_GPU_INC=/usr/local/MATLAB/R2017a/toolbox/distcomp/gpu/extern/include
CUDA_INC=/usr/local/cuda/include
CUDA_LIB=/usr/local/cuda/lib64

SPDLOG_DIR=../include

CPPFLAGS+=-DNDEBUG -I$(SPDLOG_DIR)
CXXFLAGS+=-std=c++11 -Wall -Wextra -pthread
NVCCFLAGS=-std=c++11 --gpu-architecture=sm_61

all: $(TARGET)

quantilenorm_cuda.mexa64: $(OBJS) cuda-utils/libcudautils.a mex-utils/libmexutils.a utils/libutils.a
	mex -output $@ $(OBJS) -L./cuda-utils -lcudautils -L./mex-utils -lmexutils -L./utils -lutils -lmwgpu -L$(CUDA_LIB) -lcudart -ltbb
	@cp -a quantilenorm_cuda.mexa64 ../

.cpp.o: $(SRCS)
	mex -c $< $(CPPFLAGS) -I$(MATLAB_GPU_INC) -I$(CUDA_INC)

cuda-utils/libcudautils.a: cuda-utils/*.cu cuda-utils/*.h
	make -C cuda-utils

mex-utils/libmexutils.a: mex-utils/*.cpp mex-utils/*.h
	make -C mex-utils

utils/libutils.a: utils/*.cpp utils/*.h
	make -C utils

clean:
	rm -f *.mexa64 *.o
	@make -C cuda-utils clean
	@make -C mex-utils clean
	@make -C utils clean

