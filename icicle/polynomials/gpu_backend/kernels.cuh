#pragma once
#include "stdint.h"

namespace polynomials {

  template <typename E>
  __global__ void AddSubKernel(E* a_vec, E* b_vec, int a_len, int b_len, bool add1_sub0, E* result)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= max(a_len, b_len)) return;

    E a = tid >= a_len ? E::zero() : a_vec[tid];
    E b = tid >= b_len ? E::zero() : b_vec[tid];
    result[tid] = add1_sub0 ? a + b : a - b;
  }

  // assuming 1 thread. TODO: parallellize
  template <typename E>
  __global__ void HighestNonZeroIdx(E* vec, int len, int32_t* idx)
  {
    *idx = -1;
    for (int i = len - 1; len >= 0; --i) {
      if (vec[i] != E::zero()) {
        *idx = i;
        return;
      }
    }
  }
} // namespace polynomials