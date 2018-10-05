SOURCES := $(wildcard *.cpp)
OBJS := $(SOURCES:.cpp=.o)

MATLAB_INC=/usr/local/MATLAB/R2018a/extern/include

all: libmexutils.a

libmexutils.a: $(OBJS)
	ar crs $@ $(OBJS)

.cpp.o: $(SOURCES)
	g++ -std=c++11 -fPIC -c -o $*.o $< -I$(MATLAB_INC)

clean:
	@rm *.o *.a

