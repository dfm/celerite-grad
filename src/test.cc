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

template <typename matrix, typename vector>
typename matrix::Scalar compute_likelihood (
  const vector& a, const matrix& U, const matrix& V, const matrix& P, const vector& y
) {
  typedef typename matrix::Scalar T;
  const int J = matrix::ColsAtCompileTime;
  int N = a.rows();

  vector d, z;
  matrix W;
  Matrix<T, J, J> S(J, J), bS(J, J);
  Matrix<T, J, 1> F(J), G(J), bF(J), bG(J);

  S.setZero();
  d = a;
  W = V;
  int flag = celerite::factor(U, P, d, W, S);
  T ll = log(d.array()).sum();

  z = y;
  celerite::solve(U, P, d, W, z, F, G);
  ll += y.transpose() * z;

  return ll;
}

template <typename matrix, typename vector>
typename matrix::Scalar compute_grad_likelihood (
  const vector& a, const matrix& U, const matrix& V, const matrix& P, const vector& y,
  vector& ba, matrix& bU, matrix& bV, matrix& bP, vector& by
) {
  typedef typename matrix::Scalar T;
  const int J = matrix::ColsAtCompileTime;
  int N = a.rows();

  vector d, z, bd(N), bz(N);
  matrix W, bW(N, J);
  Matrix<T, J, J> S(J, J), bS(J, J);
  Matrix<T, J, 1> F(J), G(J), bF(J), bG(J);

  S.setZero();
  d = a;
  W = V;
  int flag = celerite::factor(U, P, d, W, S);
  if (flag) std::cerr << flag << std::endl;
  T ll = log(d.array()).sum();

  z = y;
  celerite::solve(U, P, d, W, z, F, G);
  ll += y.transpose() * z;

  // Seed gradients.
  bz = y;
  //by = z;
  by.setZero();
  bd.array() = 1.0 / d.array();

  bF.setZero();
  bG.setZero();

  bU.setZero();
  bP.setZero();
  bW.setZero();

  celerite::solve_grad(U, P, d, W, z, F, G, bz, bF, bG, bU, bP, bd, bW, by);
  by.array() += z.array();

  bS.setZero();
  ba.setZero();
  bV.setZero();
  celerite::factor_grad(U, P, d, W, S, bd, bW, bS, ba, bU, bV, bP);

  return ll;
}

#define NUMERICAL_GRAD(ARG, BARG)                         \
    ARG += eps;                                           \
    plus = compute_likelihood(a, U, V, P, y);             \
    ARG -= 2*eps;                                         \
    minus = compute_likelihood(a, U, V, P, y);            \
    ARG += eps;                                           \
    error = std::abs(BARG - 0.5 * (plus - minus) / eps);  \
    if (error > max_error) {                              \
      max_error = error;                                  \
      max_error_name = #ARG;                              \
    }

template <typename T, int J>
void run_test (int N) {
  const auto Options = J == 1 ? ColMajor : RowMajor;
  typedef Matrix<T, Dynamic, J, Options> matrix;
  typedef Matrix<T, Dynamic, 1> vector;

  // Random matrices
  srand(1234);
  vector a(N), y = vector::Random(N);
  matrix U = matrix::Random(N, J),
         V = matrix::Random(N, J),
         P = matrix::Random(N-1, J);
  a.setConstant(10*J);

  // Gradients
  vector ba(N), by(N);
  matrix bU(N, J),
         bV(N, J),
         bP(N-1, J);

  compute_grad_likelihood(a, U, V, P, y, ba, bU, bV, bP, by);

  T eps = T(1e-8);
  T plus, minus, error, max_error = T(0.0);
  auto max_error_name = "nothing";
  for (int n = 0; n < N; ++n) {
    NUMERICAL_GRAD(a(n), ba(n));
    NUMERICAL_GRAD(y(n), by(n));
    for (int j = 0; j < J; ++j) {
      NUMERICAL_GRAD(U(n, j), bU(n, j));
      NUMERICAL_GRAD(V(n, j), bV(n, j));
      if (n < N-1) {
        NUMERICAL_GRAD(P(n, j), bP(n, j));
      }
    }
  }

  std::cout << max_error << std::endl;
  std::cout << max_error_name << std::endl;
  //std::cout << bU2.col(0).transpose() << std::endl;
  //std::cout << (bU.col(0) - bU2.col(0)).transpose() << std::endl;
}

int main ()
{
  run_test<double, 2>(10);

  return 0;
}
