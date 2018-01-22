#include <iostream>
#include <Eigen/Core>
#include <sys/time.h>

#include "celerite.h"

using namespace Eigen;

double get_timestamp () {
  struct timeval now;
  gettimeofday (&now, NULL);
  return double(now.tv_usec) * 1.0e-6 + double(now.tv_sec);
}

template <typename T, int J_comp>
void run_benchmark (int J, int N) {
  const auto Options = J_comp == 1 ? ColMajor : RowMajor;
  typedef Matrix<T, Dynamic, J_comp, Options> matrix;
  typedef Matrix<T, Dynamic, 1> vector;

  // Random matrices
  srand(1234);
  vector A(N), Y = vector::Random(N), Z, bA(N), bD(N), bY(N), bZ(N);
  matrix U = matrix::Random(N, J), bU(N, J),
         V0 = matrix::Random(N, J), V(N, J), bV(N, J), bW(N, J),
         P = matrix::Random(N-1, J), bP(N-1, J);
  Matrix<T, J_comp, J_comp, Options> S(J, J), bS(J, J);
  Matrix<T, J_comp, 1> F(J), G(J), bF(J), bG(J);
  S.setZero();

  // Likelihood time
  double strt, end, count = 0.0;

  strt = get_timestamp();
  do {
    V << V0;
    A.setConstant(10*J);
    int flag = celerite::factor(U, P, A, V, S);
    T ll = log(A.array()).sum();

    Z = Y;
    celerite::solve(U, P, A, V, Z, F, G);
    ll += Y.transpose() * Z;

    end = get_timestamp();
    count += 1.0;
  } while ((end - strt < 0.7) && (count < 3.0));
  std::cout << sizeof(T) << "," << J_comp << "," << J << "," << N << ",";
  std::cout << ((end - strt) / count) << ",";

  // Grad time
  strt = get_timestamp();
  count = 0.0;
  do {
    bZ = Y;
    bF.setZero();
    bG.setZero();
    celerite::solve_grad(U, P, A, V, Z, F, G, bZ, bF, bG, bU, bP, bD, bW, bY);
    bY.array() += Z.array();

    bD.array() = 1.0 / A.array();
    bW.setZero();
    bS.setZero();
    celerite::factor_grad(U, P, A, V, S, bD, bW, bS, bA, bU, bV, bP);
    end = get_timestamp();
    count += 1.0;
  } while ((end - strt < 0.7) && (count < 3.0));

  std::cout << ((end - strt) / count);
  std::cout << std::endl;
}

#define RUN_BENCHMARK(J, N)      \
  run_benchmark<double, J>(J, N);      \
  run_benchmark<double, Dynamic>(J, N);

#define RUN_BENCHMARKS(J)        \
RUN_BENCHMARK(J, 64    )         \
RUN_BENCHMARK(J, 128   )         \
RUN_BENCHMARK(J, 256   )         \
RUN_BENCHMARK(J, 512   )         \
RUN_BENCHMARK(J, 1024  )         \
RUN_BENCHMARK(J, 2048  )         \
RUN_BENCHMARK(J, 4096  )         \
RUN_BENCHMARK(J, 8192  )         \
RUN_BENCHMARK(J, 16384 )         \
RUN_BENCHMARK(J, 32768 )         \
RUN_BENCHMARK(J, 65536 )         \
RUN_BENCHMARK(J, 131072)         \
RUN_BENCHMARK(J, 262144)


int main ()
{
  std::cout << "size,J_comp,J,N,time,grad_time\n";
  //RUN_BENCHMARKS(1);
  RUN_BENCHMARKS(2);
  RUN_BENCHMARKS(4);
  RUN_BENCHMARKS(8);
  RUN_BENCHMARKS(16);
  RUN_BENCHMARKS(32);
  RUN_BENCHMARKS(64);
  RUN_BENCHMARKS(128);

  return 0;
}
