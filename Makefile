default:
	${CXX} -o bin/benchmark src/benchmark.cc -Isrc -I/usr/local/include/eigen3 -O2 -march=native -std=c++14 -DNDEBUG
