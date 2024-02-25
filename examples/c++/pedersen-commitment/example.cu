#include <iostream>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <nvml.h>

#define CURVE_ID 1
#include "curves/curve_config.cuh"
#include "appUtils/msm/msm.cu"

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
  return r;
}

// Check if y2 is a quadratic residue using Euler's Criterion
bool quadratic_residue(T y2) {
  return modPow(y2, T::div2(T::zero() - T::one())) == T::one();
}

// modular square root adapted from:
// https://github.com/ShahjalalShohag/code-library/blob/main/Number%20Theory/Tonelli%20Shanks%20Algorithm.cpp
bool mySQRT(T a, T *result) {
  // a %= p; if (a < 0) a += p;
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

void point_near_x(T x, affine_t *point) {
  const T wb = T { weierstrass_b };
  T y2;
  while (y2 = x*x*x + wb, quadratic_residue(y2) == false)
  {
    x = x + T::one();
  };
  T y;
  bool found = mySQRT(y2, &y);
  assert(y*y == y2);
  point->x = x;
  point->y = y;
}

static int seed = 0;
static HOST_INLINE T rand_host_seed()
  {
    std::mt19937_64 generator(seed++);
    std::uniform_int_distribution<unsigned> distribution;
    
    T value;
    for (unsigned i = 0; i <  T::TLC-1 ; i++)
    // TODO: use the full range of limbs: for (unsigned i = 0; i <  T::TLC ; i++)
      value.limbs_storage.limbs[i] = distribution(generator);
    // while (lt(Field{get_modulus()}, value))
    //   value = value - Field{get_modulus()};
    return value;
  }

using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg) printf("%s: %.0f ms\n", msg, FpMilliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());


int main(int argc, char** argv)
{
  const unsigned N = pow(2, 10);
  T xs[N];
  START_TIMER(gen);
  // T::RandHostMany(xs, N);
  seed = 1234;
  for (unsigned i = 0; i < N; i++) {
    xs[i] = rand_host_seed();
  }
  END_TIMER(gen, "Generated random x");
  std::cout << "rand_host_seed 0: " << xs[0]  << std::endl;
  std::cout << "rand_host_seed 1: " << xs[1]  << std::endl;
  
  affine_t points[N];
  START_TIMER(points);
  for (unsigned i = 0; i < N; i++) {
    point_near_x(xs[i], &points[i]);
  }
  END_TIMER(points, "Generated points");
  
  std::cout << "Using MSM scalars to store commitment vector" << std::endl;
  projective_t result;
  scalar_t* scalars = new scalar_t[N];
  scalar_t::RandHostMany(scalars, N);

  std::cout << "Executing MSM" << std::endl;
  auto config = msm::DefaultMSMConfig<scalar_t>();
  msm::MSM<scalar_t, affine_t, projective_t>(scalars, points, N, config, &result);

  std::cout << "Cleaning up..." << std::endl;
  delete[] scalars;
  return 0;
}
