#include <iostream>
#define CURVE_ID BN254
#include "curves/curve_config.cuh"
#include "polynomials/polynomials.cpp"
#include "appUtils/ntt/ntt.cuh"
using namespace curve_config;
using namespace polynomials;

typedef Polynomial<scalar_t> Polynomial_t;

void example1() {
  std::cout << "Polynomial evaluation on random value" << std::endl;
  const scalar_t coeffs[3] = {scalar_t::one(), scalar_t::from(2), scalar_t::from(3)};
  auto f = Polynomial_t::from_coefficients(coeffs, 3);
  std::cout << "f = " << f << std::endl;
  scalar_t x = scalar_t::rand_host();
  std::cout << "x = " << x << std::endl;
  auto f_x = f(x); // evaluation
  std::cout << "f(x) = " << f_x << std::endl;
}

void example2(int size) {
  std::cout << "Polynomial evaluation on roots of unity" << std::endl;
  // const int size = 100;
  const int log_size = (int)ceil(log2(size));
  const int nof_evals = 1 << log_size;
  auto coeff = std::make_unique<scalar_t[]>(size);
  for (int i = 0; i < size; i++)
      coeff[i] = scalar_t::rand_host();
  auto f = Polynomial_t::from_coefficients(coeff.get(), size);

  // root of unity
  auto omega = scalar_t::omega(log_size);
  scalar_t evals[nof_evals] = {scalar_t::zero()};
  auto x = scalar_t::one();
  for (int i = 0; i < nof_evals; ++i) {
    evals[i] = f(x);
    x = x * omega;
  }
  // reconstruct f from evaluations
  auto fr = Polynomial_t::from_rou_evaluations(evals, nof_evals);
  // make sure they are equal, that is f-fr=0
  auto h = f - fr;
  

  std::cout << "degree of f - fr = " << h.degree() << std::endl;
}

int main(int argc, char** argv)
{
  // const static auto one = scalar_t::one();
  // const static auto two = scalar_t::from(2);
  // const static auto three = scalar_t::from(3);
  // init NTT domain: TODO: can we hide this in the library?
  static const int MAX_NTT_LOG_SIZE = 24;
  auto ntt_config = ntt::DefaultNTTConfig<scalar_t>();
  const scalar_t basic_root = scalar_t::omega(MAX_NTT_LOG_SIZE);
  ntt::InitDomain(basic_root, ntt_config.ctx);
    
  // initializing polynomimals factory for CUDA backend
  Polynomial_t::initialize(std::make_unique<CUDAPolynomialFactory<>>());


  example1();
  // std::cout << "Polynomial evaluation on random value" << std::endl;
  // const scalar_t coeffs[3] = {one, two, three};
  // auto f = Polynomial_t::from_coefficients(coeffs, 3);
  // std::cout << "f = " << f << std::endl;
  // scalar_t x = scalar_t::rand_host();
  // std::cout << "x = " << x << std::endl;
  // auto f_x = f(x); // evaluation
  // std::cout << "f(x) = " << f_x << std::endl;

  
  example2(100);

  return 0;
}