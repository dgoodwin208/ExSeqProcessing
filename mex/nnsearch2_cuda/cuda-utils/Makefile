.SUFFIXES: .cu
.PHONY: clean

SOURCES := $(wildcard *.cu)
HEADERS := $(wildcard *.h)
OBJS := $(SOURCES:.cu=.o)
TEST_OBJS := $(patsubst %.cu,test/%.o,$(SOURCES))

NVCC_FLAGS=--gpu-architecture=sm_61
SPDLOG_INCLUDE=../../include

all: libcudautils.a libcudautils_test.a

libcudautils.a: $(OBJS)
	ar crs $@ $(OBJS)

libcudautils_test.a: test $(TEST_OBJS)
	ar crs $@ $(TEST_OBJS)

.cu.o: $(SOURCES) $(HEADERS)
	nvcc -std=c++11 $(NVCC_FLAGS) -Xcompiler -fPIC -c -o $*.o $< -I$(SPDLOG_INCLUDE) -I..

test/%.o: %.cu
	nvcc -g -std=c++11 $(NVCC_FLAGS) -D DEBUG_DIST_CHECK -Xcompiler -fPIC -c -o $@ $< -I$(SPDLOG_INCLUDE) -I..

test:
	mkdir -p test

clean:
	@rm -f *.o *.a test/*.o

