CC = gcc
CXXFLAGS  = -g -Wall -O3

all: symmetric directed
symmetric:
	$(CXX) $(CXXFLAGS) -o symmetric src/symmetric.cpp

directed:
	$(CXX) $(CXXFLAGS) -o directed src/directed.cpp

.PHONY: clean

clean:
	rm -f symmetric
	rm -f directed
