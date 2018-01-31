TARGET=semaphore

SOURCES := $(wildcard *.cpp)
OBJS := $(SOURCES:.cpp=.o)

all: $(TARGET)

semaphore: $(OBJS)
	g++ -o $@ $(OBJS) -lpthread

.cpp.o: $(SOURCES)
	g++ -c $< -std=c++11

clean:
	-rm $(TARGET) $(OBJS)

