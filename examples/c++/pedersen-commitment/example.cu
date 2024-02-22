#include <iostream>
#include <iomanip>
#include <chrono>
#include <nvml.h>

#define CURVE_ID 1
#include "curves/curve_config.cuh"
// #include "utils/device_context.cuh"
// #include "utils/vec_ops.cu"

using namespace curve_config;



scalar_t modPow(scalar_t base, scalar_t exp) {
    scalar_t r = scalar_t::one();
    scalar_t b = base;
    scalar_t e = exp;
    while (e != scalar_t::zero()) {
        // If exp is odd, multiply the base with result
        if (scalar_t::is_odd(e)) {
            r = r * b;
        }
        // Now exp must be even, divide it by 2
        e =scalar_t::div2(e);
        b = b * b;
    }
    // *result = r;
    return r;
}

 

bool mySQRT(scalar_t a, /*scalar_t p,*/ scalar_t *result) {
  // a %= p; if (a < 0) a += p;
  if (a == scalar_t::zero()) {
    *result = scalar_t::zero();
    return true;
  }
  if (modPow(a, scalar_t::div2(scalar_t::zero()-scalar_t::one())) != scalar_t::one() ) {
    return false; // solution does not exist
  }
  // if (p % 4 == 3) return power(a, (p + 1) / 4, p);
  scalar_t s = scalar_t::zero()-scalar_t::one(); // p - 1, 
  scalar_t n = scalar_t::one() + scalar_t::one(); //2;
  scalar_t r = scalar_t::zero(); 
  scalar_t m;
  while (scalar_t::is_even(s)) {
    r = r + scalar_t::one();
    s = scalar_t::div2(s); //s /= 2;
  }
  // find a non-square mod p
  // while (power(n, (p - 1) / 2, p) != p - 1) ++n;
  while (modPow(n, scalar_t::div2((scalar_t::zero() - scalar_t::one())) ) != scalar_t::zero() - scalar_t::one()) {
    n = n + scalar_t::one();
  }


  // int x = power(a, (s + 1) / 2, p);
  scalar_t x = modPow(a, scalar_t::div2(s + scalar_t::one()));
  //int b = power(a, s, p)
  scalar_t b = modPow(a, s);
  //int g = power(n, s, p);
  scalar_t g = modPow(n, s);
  for (;; r = m) {
    // int t = b;
    scalar_t t = b;
    // for (m = 0; m < r && t != 1; ++m) t = 1LL * t * t % p;
    for (m = scalar_t::zero(); scalar_t::lt(m,r) /* m < r*/ && t != scalar_t::one(); m = m + scalar_t::one()) t =  t * t;

    if (m == scalar_t::zero() ) {
      *result = x;
      return true;
    }
    //int gs = power(g, 1LL << (r - m - 1), p);
    scalar_t gs = modPow(g, modPow(scalar_t::one()+scalar_t::one(), r - m - scalar_t::one()) );

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
  scalar_t base = scalar_t::rand_host();
  std::cout << base << std::endl;
  scalar_t exp = scalar_t::div2(scalar_t::zero()-scalar_t::one());
  std::cout << exp << std::endl;
  scalar_t result = modPow(base, exp);
  std::cout << "Result: " << result << std::endl;
  scalar_t a = scalar_t::zero();
  int b=0;  
  for(int i=0; i<4; i++) {
    a = a + scalar_t::one();
    b = b + 1;
  }
  std::cout << b << " " << SQRT(b,23) << std::endl;
  std::cout << "test lt " << scalar_t::lt(scalar_t::one(),scalar_t::one()) << std::endl;
  scalar_t res;
  bool found = mySQRT(a, &res);
  std::cout << a << " " << found << " " << res << std::endl;
  return 0;
}
