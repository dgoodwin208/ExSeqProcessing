GTEST_DIR = /mp/nas1/share/lib/googletest/googletest

CPPFLAGS += -std=c++11 -isystem $(GTEST_DIR)/include
CXXFLAGS += -g -Wall -Wextra -pthread -O3
NVCC = /usr/local/cuda/bin/nvcc
#NVCC_FLAGS = -g -fPIC -arch=sm_61 -c++=11

CUDA_INC = /usr/local/cuda/include
CUDA_LIB = /usr/local/cuda/lib64

SPDLOG_DIR = ../../include

UTILS_DIR = ../utils
CUDAUTILS_DIR = ../cuda-utils

TESTS = gtest_utils gtest_cudautils  

GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h


all : $(TESTS)

clean :
	rm -f $(TESTS) *.o

gtest_utils.o : gtest_utils.cpp $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(UTILS_DIR) -c gtest_utils.cpp

gtest_utils : gtest_utils.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(GTEST_DIR)/libgtest_main.a $(GTEST_DIR)/libgtest.a -lpthread $^ -o $@ -L$(UTILS_DIR) -lutils

gtest_cudautils.o : gtest_cudautils.cpp $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) -c gtest_cudautils.cpp -L$(CUDA_LIB) -I$(CUDA_INC) -I$(CUDAUTILS_DIR) -I$(SPDLOG_DIR)

gtest_cudautils : gtest_cudautils.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(GTEST_DIR)/libgtest_main.a $(GTEST_DIR)/libgtest.a $(CUDAUTILS_DIR)/libcudautils.a -I$(CUDA_INC) -I$(CUDAUTILS_DIR) -L$(CUDAUTILS_DIR) -L$(CUDA_LIB) -lcudart -lcudautils -lpthread -o $@ 

