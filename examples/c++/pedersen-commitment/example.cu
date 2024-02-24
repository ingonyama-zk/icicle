#include <iostream>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <nvml.h>

#define CURVE_ID 1
#include "curves/curve_config.cuh"

// #include "utils/device_context.cuh"
// #include "utils/vec_ops.cu"

using namespace curve_config;


typedef point_field_t T;


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
    // *result = r;
    return r;
}

// def is_quadratic_residue(x, p, a=0, b=3):
//     # Calculate the RHS of the curve equation
//     rhs = (x**3 + a*x + b) % p
    
//     # Check if rhs is a quadratic residue mod p using Euler's Criterion
//     return pow(rhs, (p-1) // 2, p) == 1

// Check if y2 is a quadratic residue using Euler's Criterion
bool quadratic_residue(T y2) {
  return modPow(y2, T::div2(T::zero() - T::one())) == T::one();
}

bool mySQRT(T a, T *result) {
  // a %= p; if (a < 0) a += p;
  if (a == T::zero()) {
    *result = T::zero();
    return true;
  }
  if (modPow(a, T::div2(T::zero() - T::one())) != T::one() ) {
    return false; // solution does not exist
  }
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
  // while (power(n, (p - 1) / 2, p) != p - 1) ++n;
  while (modPow(n, T::div2((T::zero() - T::one())) ) != T::zero() - T::one()) {
    n = n + T::one();
  }


  // int x = power(a, (s + 1) / 2, p);
  T x = modPow(a, T::div2(s + T::one()));
  //int b = power(a, s, p)
  T b = modPow(a, s);
  //int g = power(n, s, p);
  T g = modPow(n, s);
  for (;; r = m) {
    // int t = b;
    T t = b;
    // for (m = 0; m < r && t != 1; ++m) t = 1LL * t * t % p;
    for (m = T::zero(); T::lt(m,r) /* m < r*/ && t != T::one(); m = m + T::one()) t =  t * t;

    if (m == T::zero() ) {
      *result = x;
      return true;
    }
    //int gs = power(g, 1LL << (r - m - 1), p);
    T gs = modPow(g, modPow(T::one() + T::one(), r - m - T::one()) );

    g = gs * gs ;
    x = x * gs ;
    b =  b * g ;
  }
}





int power(long long n, long long k, const int mod) {
  int ans = 1 % mod; n %= mod; if (n < 0) n += mod;
  while (k) {
    if (k & 1) ans = (long long) ans * n % mod;
    n = (long long) n * n % mod;
    k >>= 1;
  }
  return ans;
}

int SQRT(int a, int p) {
  a %= p; if (a < 0) a += p;
  if (a == 0) return 0;
  if (power(a, (p - 1) / 2, p) != 1) return -1; // solution does not exist
  if (p % 4 == 3) return power(a, (p + 1) / 4, p);
  int s = p - 1, n = 2;
  int r = 0, m;
  while (s % 2 == 0) ++r, s /= 2;
  // find a non-square mod p
  while (power(n, (p - 1) / 2, p) != p - 1) ++n;
  int x = power(a, (s + 1) / 2, p);
  int b = power(a, s, p), g = power(n, s, p);
  for (;; r = m) {
    int t = b;
    for (m = 0; m < r && t != 1; ++m) t = 1LL * t * t % p;
    if (m == 0) return x;
    int gs = power(g, 1LL << (r - m - 1), p);
    g = 1LL * gs * gs % p;
    x = 1LL * x * gs % p;
    b = 1LL * b * g % p;
  }
}


int main(int argc, char** argv)
{
  T x = T::rand_host();
  const T wb = T { weierstrass_b };
  // const scalar_t wb = scalar_t::one()+scalar_t::one()+scalar_t::one(); // 3
  T y2;
  
  while (y2 = x*x*x + wb, quadratic_residue(y2) == false)
  {
    std::cout << "x: " << x << std::endl;
    x = x + T::one();
  }
  std::cout << "x: " << x << " y2: " << y2 << std::endl;
  T y;
  bool found = mySQRT(y2, &y);
  std::cout << "x: " << x << " " << found << " y: " << T::zero()- y << std::endl;
  std::cout << "y*y: " << y*y << " y2: " << y2 << std::endl;
  assert(y*y == y2);
  // affine_t pt = affine_t{ x, y }; 
  affine_t pt;
  pt.x = x;
  pt.y = y;
  return 0;

  

  

  // for(int i=0; i<16; i++) {
  //   a = a + scalar_t::one();
  //   b = b + 1;
  // }
  // std::cout << b << " " << SQRT(b,23) << std::endl;
  // std::cout << "test modPow " << modPow(scalar_t::one()+scalar_t::one(), scalar_t::zero())   << std::endl;
  
  return 0;
}
