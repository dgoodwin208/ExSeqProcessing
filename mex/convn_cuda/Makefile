TARGET=lib-cufft-utils lib-tests convn_cuda.o convn_1GPU_cuda.o fft3_cuda.o convn_1GPU_cuda.mexa64 convn_cuda.mexa64 fft3_cuda.mexa64 # lib-cuda-utils lib-cudnn-utils convn_arrayfire.mexa64 convn_arrayfire_cuda.mexa64 lib-tests 

SOURCES := $(wildcard convn_cuda.cpp fft3_cuda.cpp)
OBJS := $(SOURCES:.cpp=.o)

MATLAB_GPU_INC=/usr/local/MATLAB/R2017a/toolbox/distcomp/gpu/extern/include
CUDA_INC=/usr/local/cuda/include
CUDA_LIB=/usr/local/cuda/lib64
CUDNN_DIR=cudnn-utils
CUFFT_DIR=cufft-utils

all: $(TARGET)

convn_cuda.mexa64: convn_cuda.o $(CUFFT_DIR)/libcufftutils.a
	mex -compatibleArrayDims -outdir ../ -output $@ $^ $(CUFFT_DIR)/libcufftutils.a -lmwgpu -L$(CUDA_LIB) -I$(CUDA_INC) -lcufft -lcudart

convn_1GPU_cuda.mexa64: convn_1GPU_cuda.o $(CUFFT_DIR)/libcufftutils.a
	mex -compatibleArrayDims -outdir ../ -output $@ $^ $(CUFFT_DIR)/libcufftutils.a -lmwgpu -L$(CUDA_LIB) -I$(CUDA_INC) -lcufft -lcudart

fft3_cuda.mexa64: fft3_cuda.o $(CUFFT_DIR)/libcufftutils.a
	mex -compatibleArrayDims -outdir ../ -output $@ $^ $(CUFFT_DIR)/libcufftutils.a -lmwgpu -L$(CUDA_LIB) -I$(CUDA_INC) -lcufft -lcudart

fft3_cuda.o: fft3_cuda.cpp
	mex -compatibleArrayDims -c $^ -I../include -I$(MATLAB_GPU_INC) -I$(CUDA_INC) -I$(CUFFT_DIR)

convn_cuda.o: convn_cuda.cpp
	mex -compatibleArrayDims -c $^ -I../include -I$(MATLAB_GPU_INC) -I$(CUDA_INC) -I$(CUFFT_DIR)

convn_1GPU_cuda.o: convn_1GPU_cuda.cpp
	mex -compatibleArrayDims -c $^ -I../include -I$(MATLAB_GPU_INC) -I$(CUDA_INC) -I$(CUFFT_DIR)

lib-cufft-utils:
	make -C cufft-utils

#.cpp.o: $(SOURCES)
	#mex -compatibleArrayDims -c $< -I../include -I$(MATLAB_GPU_INC) -I$(CUDA_INC) -I$(CUFFT_DIR)
	##mex -compatibleArrayDims -c $< -I../include -I$(MATLAB_GPU_INC) -I$(CUDA_INC) -I$(CUDNN_DIR)

# convn_cuda.mexa64: $(OBJS) $(CUDNN_DIR)/libcudnnutils.a
#	mex -compatibleArrayDims -outdir ../ -output $@ $(OBJS) $(CUDNN_DIR)/libcudnnutils.a -lmwgpu -L$(CUDA_LIB) -lcudart -lcudnn #-ltbb 

#convn_arrayfire_cuda.mexa64: convn_arrayfire_cuda.cpp
	#mex -compatibleArrayDims -outdir ../ -output $@ $^ -I$(CUDA_INC) -I/usr/local/include -L$(CUDA_LIB) -lcudart -L/usr/local/lib -lafcuda
	#mex -compatibleArrayDims -outdir ../ -output $@ $^ -I$(CUDA_INC) -I/usr/local/include -L$(CUDA_LIB) -lcudart -L/usr/local/lib -laf -lafcuda

#convn_arrayfire.mexa64: convn_arrayfire.cpp
	#mex -g -compatibleArrayDims -outdir ../ -output $@ $^ -I/usr/local/include -L/usr/local/lib -lafcpu

#lib-cuda-utils:
#	make -C cuda-utils

#lib-cudnn-utils:
#	make -C cudnn-utils libcudnnutils.a

lib-tests:
	make -C tests

clean:
	-rm *.mexa64 *.o
	-rm ../convn_cuda.mexa64
	-rm ../convn_1GPU_cuda.mexa64
	-rm ../fft3_cuda.mexa64
	@make -C tests clean
	@make -C cufft-utils clean
	#-rm ../convn_cuda.mexa64
	#-rm ../convn_arrayfire.mexa64
	#@make -C cuda-utils clean
	#@make -C cudnn-utils clean

