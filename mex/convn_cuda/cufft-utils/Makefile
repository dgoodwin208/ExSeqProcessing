NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall --gpu-architecture=sm_61
CUFFTLIB = -L/usr/local/cuda/lib -lcufft -I/usr/local/cuda/inc

main.exe: Multi_GPU_FFT_check.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(CUFFTLIB)

clean:
	rm -f *.o *.exe
