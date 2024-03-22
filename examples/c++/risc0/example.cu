
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


// modular power
T modPow(T base, T exp) {
  T r = T::one();
  T b = base;
  T e = exp;
  while (e != T::zero()) {
      // If exp is odd, multiply the base with result
      if (T::is_odd(e)) {
          r = r * b;
      }
      // Now exp must be even, divide it by 2
      e =T::div2(e);
      b = b * b;
  }
  return r;
}

// modular square root adapted from:
// https://github.com/ShahjalalShohag/code-library/blob/main/Number%20Theory/Tonelli%20Shanks%20Algorithm.cpp
bool mySQRT(T a, T *result) {
  if (a == T::zero()) {
    *result = T::zero();
    return true;
  }
  if (modPow(a, T::div2(T::zero() - T::one())) != T::one() ) {
    return false; // solution does not exist
  }
  // TODO: consider special cases
  // if (p % 4 == 3) return power(a, (p + 1) / 4, p); 
  T s = T::zero() - T::one(); // p - 1, 
  T n = T::one() + T::one(); //2;
  T r = T::zero(); 
  T m;
  while (T::is_even(s)) {
    r = r + T::one();
    s = T::div2(s); //s /= 2;
  }
  // find a non-square mod p
  while (modPow(n, T::div2((T::zero() - T::one())) ) != T::zero() - T::one()) {
    n = n + T::one();
  }
  T x = modPow(a, T::div2(s + T::one()));
  T b = modPow(a, s);
  T g = modPow(n, s);
  for (;; r = m) {
    T t = b;
    for (m = T::zero(); T::lt(m,r) /* m < r*/ && t != T::one(); m = m + T::one()) t =  t * t;
    if (m == T::zero() ) {
      *result = x;
      return true;
    }
    T gs = modPow(g, modPow(T::one() + T::one(), r - m - T::one()) );
    g = gs * gs ;
    x = x * gs ;
    b =  b * g ;
  }
}


int main(int argc, char** argv)
{

    // max number: modulus - 1
    scalar_t pm1 = scalar_t::zero()-scalar_t::one();
    // int j = fp_config::limbs_count;
    std::cout << pm1 << std::endl;
    int logn=3;
    int n = 1 << logn;
    scalar_t sn = scalar_t::from(n);
    scalar_t snm1 = sn - scalar_t::one();
    T xi = pm1 * scalar_t::inverse(sn);
    scalar_t w = scalar_t::omega(logn);
    scalar_t w2 = scalar_t::omega(1+logn);
    std::cout << "w: " << w << std::endl;
    std::cout << "w2: " << w2 << std::endl;
    std::cout << "w2^2: " << w2 * w2 << std::endl;
    scalar_t ww = scalar_t::one();
    for( int i = 0; i <= n; i++) {
        
        std::cout << "i: " << i << " ww: " << ww << std::endl;
        ww = ww * w;
    }


    // Initialize NTT. TODO: can we hide this in the library?
    static const int MAX_NTT_LOG_SIZE = 24;
    auto ntt_config = ntt::DefaultNTTConfig<scalar_t>();
    const scalar_t basic_root = scalar_t::omega(MAX_NTT_LOG_SIZE);
    ntt::InitDomain(basic_root, ntt_config.ctx);

    // Virtual factory design pattern: initializing polynomimals factory for CUDA backend
    Polynomial_t::initialize(std::make_unique<CUDAPolynomialFactory<>>());
    // const int logN = 10;
    // const int N = 1 << logN;
    std::cout << "Generate trace columns" << std::endl;
    scalar_t* trace = new scalar_t[n];
    scalar_t::RandHostMany(trace, n);


    std::cout << std::endl << "Reconstruct polynomial from values at roots of unity" << std::endl;
    auto f = Polynomial_t::from_rou_evaluations(trace, n);
    auto x = scalar_t::one();
    auto omega = w;
    for (int i = 0; i < n; ++i) {
        std::cout << "i: " << f(x) << " trace: " << trace[i] << std::endl;
        x = x * omega;
    }

    
//   example_evaluate();
//   example_clone(10);
//   example_from_rou(100);
//   example_addition(12, 17);
//   example_addition_inplace(2, 2);
//   example_multiplication(15, 12);
//   example_multiplicationScalar(15);
//   example_monomials();
//   example_ReadCoeffsToHost();
//   example_divisionSmall();
//   example_divisionLarge(12, 2);
//   example_divideByVanishingPolynomial();
  
  return 0;
}