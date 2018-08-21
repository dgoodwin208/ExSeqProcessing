.SUFFIXES: .cu
.PHONY: clean

TARGET: libcudautils.a

SRCS := $(wildcard *.cpp)
OBJS := $(SRCS:.cpp=.o)
CU_SRCS := $(wildcard *.cu)
HEADERS := $(wildcard *.h)
CU_OBJS := $(CU_SRCS:.cu=.o)

CUDA_INC=/usr/local/cuda/include
SPDLOG_INC=../include

CPPFLAGS=-I$(SPDLOG_INC) -I..
CXXFLAGS=-std=c++11
NVCCFLAGS=-std=c++11 --gpu-architecture=sm_61

all: $(TARGET)

libcudautils.a: $(OBJS) $(CU_OBJS)
	ar crs $@ $^

.cu.o: $(CU_SRCS) $(HEADERS)
	nvcc $(NVCCFLAGS) $(CPPFLAGS) -Xcompiler -fPIC -c -o $*.o $<

.cpp.o: $(SRCS) $(HEADERS)
	g++ $(CXXFLAGS) $(CPPFLAGS) -fPIC -c -o $@ $<

clean:
	@rm -f *.o *.a

