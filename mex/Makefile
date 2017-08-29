TARGET=mergesortfiles.mexa64 meanfiles.mexa64 sumfiles.mexa64 substituteToNormValues.mexa64 semaphore.mexa64

all: ${TARGET}

mergesortfiles.mexa64: mergesortfiles.cpp
	mex $<

meanfiles.mexa64: meanfiles.cpp
	mex $<

sumfiles.mexa64: sumfiles.cpp
	mex $<

substituteToNormValues.mexa64: substituteToNormValues.cpp
	mex $<

semaphore.mexa64: semaphore.cpp
	mex $<

clean:
	-rm ${TARGET}

