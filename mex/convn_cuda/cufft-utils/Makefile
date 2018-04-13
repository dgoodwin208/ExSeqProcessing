OBJ= convn_impl.o # main.exe
NVCC = /usr/local/cuda/bin/nvcc 
NVCC_FLAGS = -g -Xcompiler -fPIC --gpu-architecture=sm_61 -std=c++11
CUFFTLIB = -L/usr/local/cuda/lib64 -lcufft -I/usr/local/cuda/include

all: libcufftutils.a

libcufftutils.a: $(OBJ)
	ar crs $@ convn_impl.o

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $(CUFFTLIB) -c $^ 

#main.exe: convn_impl.cu
	#$(NVCC) $(NVCC_FLAGS) $^ -g -o $@ $(CUFFTLIB)

clean:
	rm -rf *.o *.exe *.a

clobber: clean
