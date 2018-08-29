check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

OBJ = fp16_dev.o fp16_emu.o conv_sample.o
INCLUDES := -I$(CUDA_ROOT_DIR)/include
LIBS := -L$(CUDA_ROOT_DIR)/lib64 -L/usr/local/lib
LIB_FLAGS := -lcudart -lcublas -lcudnn -lstdc++ -lm
CCFLAGS := -Xcompiler -fPIC
NVCC_FLAGS=--gpu-architecture=sm_61

all: libcudnnutils.a

libcudnnutils.a: $(OBJ)
	ar crs $@ $^ 

%.o: %.cpp 
	g++ $(INCLUDES) -fPIC $(LIB_FLAGS) $(LIBS) -o $@ -c $<

%.o: %.cu
	nvcc $(INCLUDES) $(NVCC_FLAGS) $(CCFLAGS) $(GENCODE_FLAGS) $(LIB_FLAGS) $(LIBS) -o $@ -c $<

clean:
	rm -rf *o
	rm -rf *a

clobber: clean
