SOURCES := $(wildcard *.cpp)
OBJS := $(SOURCES:.cpp=.o)

SPDLOG_INCLUDE=../../include

all: libutils.a

libutils.a: $(OBJS)
	ar crs $@ $(OBJS)

.cpp.o: $(SOURCES)
	g++ -std=c++11 -fPIC -c -o $*.o $< -I$(SPDLOG_INCLUDE)

clean:
	@rm *.o *.a

