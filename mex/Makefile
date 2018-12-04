.SUFFIXES: .mexa64

TARGET=mergesortfiles.mexa64 meanfiles.mexa64 sumfiles.mexa64 substituteToNormValues.mexa64 semaphore.mexa64 quantilenorm_cuda_init.mexa64 quantilenorm_cuda_final.mexa64 sumbinfiles.mexa64

SOURCES := $(wildcard *.cpp)
OBJS := $(SOURCES:.cpp=.o)

all: $(TARGET)

.cpp.mexa64: $(SOURCES)
	mex $< -outdir bin -output $@ -I./include

clean:
	-rm $(TARGET)

