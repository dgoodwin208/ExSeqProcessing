NVCC = /usr/local/cuda/bin/nvcc 
NVCC_FLAGS = -g -std=c++11 -arch=sm_61 

all: libcudautils.a 

libcudautils.a: 3DSIFT_impl.o
	ar crs $@ 3DSIFT_impl.o 

3DSIFT_impl.o: 3DSIFT_impl.cu
	$(NVCC) $(NVCC_FLAGS) -o 3DSIFT_impl.o -c 3DSIFT_impl.cu -lcudart 

clean:
	rm -rf *.o *.exe *.a

clobber: clean
