.SUFFIXES: .cu

SOURCES := $(wildcard *.cu)
OBJS := $(SOURCES:.cu=.o)

NVCC_FLAGS=--gpu-architecture=sm_61
SPDLOG_INCLUDE=../../include

all: libcudautils.a

libcudautils.a: $(OBJS)
	ar crs $@ $(OBJS)

.cu.o: $(SOURCES)
	nvcc -std=c++11 $(NVCC_FLAGS) -Xcompiler -fPIC -c -o $*.o $< -lcudart -I$(SPDLOG_INCLUDE) -ltbb

clean:
	@rm *.o *.a

