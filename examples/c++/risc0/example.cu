
#include <iostream>
#define CURVE_ID 1
#include "curves/curve_config.cuh"
#include "polynomials/polynomials.h"
#include "polynomials/cuda_backend/polynomial_cuda_backend.cuh"
#include "appUtils/ntt/ntt.cuh"
using namespace curve_config;
using namespace polynomials;

// define the polynomial type
typedef scalar_t T;
typedef Polynomial<T> Polynomial_t;

int main(int argc, char** argv)
{

  int logn=3;
  int n = 1 << logn;

  // message domain (trace in risc0 parlance)
  scalar_t w = scalar_t::omega(logn);
  
  // codeword (extended) domain
  scalar_t w2 = scalar_t::omega(1+logn);

  // Initialize NTT
  static const int MAX_NTT_LOG_SIZE = 24;
  auto ntt_config = ntt::DefaultNTTConfig<scalar_t>();
  const scalar_t basic_root = scalar_t::omega(MAX_NTT_LOG_SIZE);
  ntt::InitDomain(basic_root, ntt_config.ctx);

  // Virtual factory design pattern: initializing polynomimals factory for CUDA backend
  Polynomial_t::initialize(std::make_unique<CUDAPolynomialFactory<>>());

  std::cout << "Generate message (trace)" << std::endl;
  scalar_t* trace = new scalar_t[n];
  scalar_t::RandHostMany(trace, n);

  std::cout << std::endl << "Reconstruct polynomial from values at roots of unity" << std::endl;
  auto f = Polynomial_t::from_rou_evaluations(trace, n);
  auto d = f.degree();
  std::cout << "Degree: " << d << std::endl;
  auto x = scalar_t::one();
  auto omega = w;
  for (int i = 0; i < n; ++i) {
    std::cout << "i: " << f(x) << " trace: " << trace[i] << std::endl;
    x = x * omega;
  }

  std::cout << std::endl << "Compute codeword at larger domain" << std::endl;
  scalar_t* codeword = new scalar_t[2*n];
  auto x2 = scalar_t::one();
  auto omega2 = w2;
  for (int i = 0; i < 2*n; ++i) {
    codeword[i] = f(x2);
    x2 = x2 * omega2;
  }
  std::cout << std::endl << "Reconstruct polynomial for the codeword" << std::endl;
  auto f2 = Polynomial_t::from_rou_evaluations(codeword, 2*n);
  auto d2 = f2.degree();
  std::cout << "Degree: " << d2 << " expected: " << d << std::endl;

  std::cout << std::endl << "Introduce error in codeword" << std::endl;
  codeword[0] = codeword[0] + scalar_t::one();
  auto f3 = Polynomial_t::from_rou_evaluations(codeword, 2*n);
  auto d3 = f3.degree();
  std::cout << "Degree: " << d3 << " was: " << d2 << std::endl;

  return 0;
}