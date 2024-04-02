
#include <iostream>
#define CURVE_ID 1
#include "curves/curve_config.cuh"
#include "polynomials/polynomials.h"
#include "polynomials/polynomials_c_api.h"
#include "polynomials/cuda_backend/polynomial_cuda_backend.cuh"
#include "appUtils/ntt/ntt.cuh"
#include "appUtils/poseidon/poseidon.cu"
#include "appUtils/tree/merkle.cu"
using namespace curve_config;
using namespace polynomials;
using namespace poseidon;
using namespace merkle;

// define the polynomial type
typedef scalar_t T;
typedef Polynomial<T> Polynomial_t;

// Merkle tree arity
#define A 2
#define T (A + 1)

int main(int argc, char** argv)
{

  const int logn=3;
  const int n = 1 << logn;

  // FRI blow-up factor: 2

  // int logblowup = 1;
  // int blowup = 1 << logblowup;

  // Finite field indexing: function omega() gives the root of unity for the given size

  // message domain (trace column in risc0 parlance)
  scalar_t w = scalar_t::omega(logn);
  
  // codeword (trace block) in expanded (2x) domain
  scalar_t w2 = scalar_t::omega(1+logn);

  // Initialize NTT
  static const int MAX_NTT_LOG_SIZE = 24;
  auto ntt_config = ntt::DefaultNTTConfig<scalar_t>();
  const scalar_t basic_root = scalar_t::omega(MAX_NTT_LOG_SIZE);
  ntt::InitDomain(basic_root, ntt_config.ctx);

  // Virtual factory design pattern: initializing polynomimals factory for CUDA backend
  Polynomial_t::initialize(std::make_unique<CUDAPolynomialFactory<>>());


  // Initialize Poseidon
  device_context::DeviceContext ctx = device_context::get_default_device_context();
  PoseidonConstants<scalar_t> constants;
  init_optimized_poseidon_constants<scalar_t>(A, ctx, &constants);

  std::cout << "Generate message (trace column)" << std::endl;
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

  std::cout << std::endl << "Interpolate codeword (trace block) at larger domain" << std::endl;
  scalar_t* codeword = new scalar_t[2*n];
  auto x2 = scalar_t::one();
  auto omega2 = w2;
  for (int i = 0; i < 2*n; ++i) {
    codeword[i] = f(x2);
    std::cout << i << " : " << codeword[i] << std::endl;
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


  std::cout << std::endl << "Commitment to the trace polynomial" << std::endl;
  scalar_t* commitment = new scalar_t[2*n];
  std::cout << "Evaluate on larger domain with a shift: " << std::endl;
  const int shift = 1; // next power of 2 for the rou
  scalar_t xs = scalar_t::omega(shift + 1 + logn);
  
  for (int i = 0; i < 2*n; ++i) {
    commitment[i] = f(xs);
    std::cout << i << " : " << commitment[i] << std::endl;
    xs = xs * omega2;
  }
  uint32_t tree_height = (logn + 1) + 1; // extra +1 for larger domain
  size_t digests_len = get_digests_len<scalar_t>(tree_height, A);
  std::cout << "Digests length: " << digests_len << std::endl;
  scalar_t* digests = new scalar_t[digests_len];
  TreeBuilderConfig config = default_merkle_config<scalar_t>();
  build_merkle_tree<scalar_t, T>(commitment, digests, tree_height, constants, config);
  std::cout << "Root: " << digests[0] << std::endl;

  std::cout << std::endl << "FRI Protocol (Commit Phase)" << std::endl;
  const int m = 2*n;
  std::cout << std::endl << "Split" << std::endl;
  scalar_t f0_coeffs[m] = {0};
  scalar_t f0even_coeffs[m/2] = {0};
  scalar_t f0odd_coeffs[m/2] = {0};
  auto f0 = f2.clone();

  

  auto cc = f0.copy_coefficients_to_host(f0_coeffs, 0, -1);
  std::cout << "Coefficients: " << cc << std::endl;
  for (int i = 0; i < m; ++i) {
    std::cout << i << ": " << f0_coeffs[i] << std::endl;
  }
  for (int i = 0; i < m/2; ++i) {
    f0even_coeffs[i] = f0_coeffs[2*i];
    f0odd_coeffs[i] = f0_coeffs[2*i+1];
    std::cout << i << ": even: " << f0even_coeffs[i] << std::endl;
    std::cout << i << ": odd:  " << f0odd_coeffs[i] << std::endl;
  }
  auto f0even = Polynomial_t::from_coefficients(f0even_coeffs, m/2);
  auto f0odd = Polynomial_t::from_coefficients(f0odd_coeffs, m/2);
  // verifier-provided randomness 
  auto r1 = scalar_t::rand_host();
  // Round 1 polynomial
  auto f1 = f0even + r1 * f0odd;
  std::cout << std::endl << "FRI Protocol (Query Phase)" << std::endl;
  std::cout << "Checking the Evaluations are Consistent" << std::endl;
  scalar_t xp = scalar_t::rand_host();
  scalar_t xm = scalar_t::zero() - xp;
  auto rhs = (r1+xp)*f0(xp)*scalar_t::inverse(scalar_t::from(2)*xp) + (r1+xm)*f0(xm)*scalar_t::inverse(scalar_t::from(2)*xm);
  auto lhs = f1(xp*xp);
  std::cout << "rhs: " << rhs << std::endl << "lhs: " << lhs << std::endl;
  if (lhs != rhs) {
    std::cout << "Error: Evaluations are not consistent" << std::endl;
  } else {
    std::cout << "Evaluations are consistent" << std::endl;
  }
  auto d1 = f1.degree();
  auto d0 = f0.degree();
  std::cout << "Degree before: " << d1 << " degree before: " << d0 << std::endl;
  return 0;
}