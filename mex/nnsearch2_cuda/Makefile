TARGET=nnsearch2_cuda.mexa64 nnsearch2_cuda

MATLAB_GPU_INC=/usr/local/MATLAB/R2018a/toolbox/distcomp/gpu/extern/include
CUDA_INC=/usr/local/cuda/include
CUDA_LIB=/usr/local/cuda/lib64

all: $(TARGET)

nnsearch2_cuda.mexa64: nnsearch2_cuda.o cuda_task_executor.o cuda-utils/libcudautils.a
	mex -output $@ $^ -L./cuda-utils -lcudautils -lmwgpu -L$(CUDA_LIB) -lcudart
	@cp -a $@ ../

nnsearch2_cuda.o: nnsearch2_cuda.cpp
	mex -c $< -I../include -I$(MATLAB_GPU_INC) -I$(CUDA_INC) -I.

.cpp.o: cmd_nnsearch2_cuda.cpp cuda_task_executor.cpp
	g++ -std=c++11 -fPIC -c -o $@ $< -I../include -I$(CUDA_INC) -I.

#lib-cuda-utils:
cuda-utils/libcudautils.a: cuda-utils/*.cu cuda-utils/*.h
	make -C cuda-utils


nnsearch2_cuda: cmd_nnsearch2_cuda.o cuda_task_executor.o cuda-utils/libcudautils.a
	g++ -o $@ $^ -L$(CUDA_LIB) -lcudart -lpthread

clean:
	-rm -f *.mexa64 *.o
	@make -C cuda-utils clean

