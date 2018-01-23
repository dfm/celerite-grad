FLAGS=-Isrc -I/usr/local/include/eigen3 -O2 -march=native -std=c++14 -DNDEBUG

default: bin/benchmark bin/test

bin/benchmark: src/benchmark.cc src/celerite.h
	${CXX} -o bin/benchmark src/benchmark.cc ${FLAGS}

bin/test: src/test.cc src/celerite.h
	${CXX} -o bin/test src/test.cc ${FLAGS}
