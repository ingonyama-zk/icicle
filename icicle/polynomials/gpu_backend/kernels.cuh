#pragma once
#include "stdint.h"

namespace polynomials {

  /*============================== add/sub ==============================*/
  template <typename T>
  __global__ void AddSubKernel(T* a_vec, T* b_vec, int a_len, int b_len, bool add1_sub0, T* result)
  {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= max(a_len, b_len)) return;

    T a = tid >= a_len ? T::zero() : a_vec[tid];
    T b = tid >= b_len ? T::zero() : b_vec[tid];
    result[tid] = add1_sub0 ? a + b : a - b;
  }

  /*============================== degree ==============================*/
  template <typename T>
  __global__ void HighestNonZeroIdx(T* vec, int len, int32_t* idx)
  {
    *idx = -1;
    for (int i = len - 1; len >= 0; --i) {
      if (vec[i] != T::zero()) {
        *idx = i;
        return;
      }
    }
  }

  /*============================== evaluate ==============================*/
  template <typename T>
  __device__ T pow(T base, int exp)
  {
    T result = T::one();
    while (exp > 0) {
      if (exp & 1) result = result * base;
      base = base * base;
      exp >>= 1;
    }
    return result;
  }

  // TODO Yuval: implement efficient reduction and support batch evaluation
  template <typename T>
  __global__ void dummyReduce(T* arr, int size, T* output)
  {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0) return;

    *output = arr[0];
    for (int i = 1; i < size; ++i) {
      *output = *output + arr[i];
    }
  }

  template <typename T>
  __global__ void evalutePolynomialWithoutReduction(T x, T* coeffs, int num_coeffs, T* tmp)
  {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid <= num_coeffs) { tmp[tid] = coeffs[tid] * pow(x, tid); }
  }

} // namespace polynomials