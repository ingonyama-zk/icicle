#pragma once
#include "stdint.h"

namespace polynomials {

  /*============================== division ==============================*/
  template <typename T>
  __global__ void school_book_division_step(T* r, T* q, const T* b, int deg_r, int deg_b, T lc_b_inv)
  {
    // computing one step 'r = r-sb' (for 'a = q*b+r') where s is a monomial such that 'r-sb' removes the highest degree
    // of r.
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t monomial = deg_r - deg_b; // monomial=1 is 'x', monomial=2 is x^2 etc.
    if (tid > deg_r) return;

    T lc_r = r[deg_r];
    T monomial_coeff = lc_r * lc_b_inv; // lc_r / lc_b
    if (tid == 0) {
      // adding monomial s to q (q=q+s)
      q[monomial] = monomial_coeff;
    }

    if (tid < monomial) return;

    T b_coeff = b[tid - monomial];
    r[tid] = r[tid] - monomial_coeff * b_coeff;
  }

} // namespace polynomials