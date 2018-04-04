.SUFFIXES: .cu
.PHONY: clean

SOURCES := $(wildcard *.cu)
OBJS := $(SOURCES:.cu=.o)
TEST_OBJS := $(patsubst %.cu,test/%.o,$(SOURCES))

NVCC_FLAGS=--gpu-architecture=sm_61
SPDLOG_INCLUDE=../../include

all: libcudautils.a libcudautils_test.a

libcudautils.a: $(OBJS)
	ar crs $@ $(OBJS)

libcudautils_test.a: test $(TEST_OBJS)
	ar crs $@ $(TEST_OBJS)

.cu.o: $(SOURCES)
	nvcc -std=c++11 $(NVCC_FLAGS) -Xcompiler -fPIC -c -o $*.o $< -lcudart -I$(SPDLOG_INCLUDE)

test/%.o: %.cu
	nvcc -std=c++11 $(NVCC_FLAGS) -D DEBUG_DIST_CHECK -Xcompiler -fPIC -c -o $@ $< -lcudart -I$(SPDLOG_INCLUDE)

test:
	mkdir -p test

clean:
	@rm -f *.o *.a test/*.o

