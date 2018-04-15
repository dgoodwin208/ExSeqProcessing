MATLAB_GPU_INC=/usr/local/MATLAB/R2017a/toolbox/distcomp/gpu/extern/include
CUDA_INC=/usr/local/cuda/include
CUDA_LIB=/usr/local/cuda/lib64
CUDA_DIR=cuda-utils

all: lib-cuda-utils KeySampleCuda.o KeySampleCuda.mexa64 
	#3DSIFT_cuda.o 3DSIFT_cuda.mexa64 lib-tests 

KeySampleCuda.mexa64: KeySampleCuda.o $(CUDA_DIR)/libcudautils.a
	mex -compatibleArrayDims -outdir ../ -output $@ $^ $(CUDA_DIR)/libcudautils.a -lmwgpu -L$(CUDA_LIB) -I$(CUDA_INC) -lcudart

KeySampleCuda.o: KeySampleCuda.cpp
	mex -compatibleArrayDims -c $^ -I../include -I$(MATLAB_GPU_INC) -I$(CUDA_INC) -I$(CUDA_DIR)

#3DSIFT_cuda.mexa64: 3DSIFT_cuda.o $(CUDA_DIR)/libcudautils.a
	#mex -compatibleArrayDims -outdir ../ -output $@ $^ $(CUDA_DIR)/libcudautils.a -lmwgpu -L$(CUDA_LIB) -I$(CUDA_INC) -lcudart

#3DSIFT_cuda.o: 3DSIFT_cuda.cpp
	#mex -compatibleArrayDims -c $^ -I../include -I$(MATLAB_GPU_INC) -I$(CUDA_INC) -I$(CUDA_DIR)

lib-cuda-utils:
	make -C cuda-utils

#lib-tests:
	#make -C tests

clean:
	-rm *.mexa64 *.o
	#-rm ../3DSIFT_cuda.mexa64
	-rm ../KeySampleCuda.mexa64
	@make -C tests clean
	@make -C cuda-utils clean
