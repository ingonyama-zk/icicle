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

  template <typename T>
  __global__ void AddSingleElementInplace(T* self, T v)
  {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0) return;
    *self = *self + v;
  }

  /*============================== degree ==============================*/
  template <typename T>
  __global__ void HighestNonZeroIdx(T* vec, int len, int32_t* idx)
  {
    *idx = -1;
    for (int i = len - 1; i >= 0; --i) {
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
    if (tid < num_coeffs) { tmp[tid] = coeffs[tid] * pow(x, tid); }
  }

  /*============================== multiply ==============================*/
  template <typename T>
  __global__ void Mul(T* element_vec1, T* element_vec2, int n, T* result)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) { result[tid] = element_vec1[tid] * element_vec2[tid]; }
  }

  /*============================== division ==============================*/
  template <typename T>
  __global__ void SchoolBookDivisionStepOnR(T* r, T* b, int r_len, int b_len, uint64_t s_monomial, T s_coeff)
  {
    // computing one step 'r = r-sb' (for 'a = q*b+r')
    // s is the highest monomial in current step. This step is subtracting the highest monomial of r. It it repeated
    // until deg(r)<deg(b)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= max(r_len, b_len)) return;
    if ((tid < s_monomial)) return;

    T b_coeff = b[tid - s_monomial];
    r[tid] = r[tid] - s_coeff * b_coeff;
  }

} // namespace polynomials