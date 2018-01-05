CXX := nvcc
TARGET := convn
CUDA_ROOT_DIR := cudnn
HEADERS := -I $(CUDA_ROOT_DIR)/include
LIBS := -L $(CUDA_ROOT_DIR)/lib64 -L/usr/local/lib
CXXFLAGS := --gpu-architecture=sm_61 -std=c++11 -O2 -lcudart -lcudnn 

all: convn

convn: $(TARGET).cu
		$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(TARGET).cu -c -o $(TARGET) 

.phony: clean

clean:
		rm $(TARGET) || echo -n ""
