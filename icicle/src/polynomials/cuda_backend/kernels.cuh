#pragma once
#include "stdint.h"

namespace polynomials {
  using namespace vec_ops;

  /*============================== add/sub ==============================*/
  template <typename T>
  __global__ void AddSubKernel(const T* a_vec, const T* b_vec, int a_len, int b_len, bool add1_sub0, T* result)
  {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= max(a_len, b_len)) return;

    T a = tid >= a_len ? T::zero() : a_vec[tid];
    T b = tid >= b_len ? T::zero() : b_vec[tid];
    result[tid] = add1_sub0 ? a + b : a - b;
  }

  // Note: must be called with 1 block, 1 thread
  template <typename T>
  __global__ void AddSingleElementInplace(T* self, T v)
  {
    *self = *self + v;
  }

  /*============================== multiplication ======================*/
  template <typename E, typename S>
  __global__ void MulScalarKernel(const E* element_vec, const S scalar, int n, E* result)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) { result[tid] = element_vec[tid] * (scalar); }
  }

  /*============================== degree ==============================*/
  template <typename T>
  __global__ void HighestNonZeroIdx(const T* vec, int len, int64_t* idx)
  {
    *idx = -1; // zero polynomial is defined with degree -1
    for (int64_t i = len - 1; i >= 0; --i) {
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
  __global__ void dummyReduce(const T* arr, int size, T* output)
  {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0) return;

    *output = arr[0];
    for (int i = 1; i < size; ++i) {
      *output = *output + arr[i];
    }
  }

  template <typename T>
  __global__ void evaluatePolynomialWithoutReduction(T x, const T* coeffs, int num_coeffs, T* tmp)
  {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_coeffs) { tmp[tid] = coeffs[tid] * pow(x, tid); }
  }

  /*============================== division ==============================*/
  template <typename T>
  __global__ void SchoolBookDivisionStep(T* r, T* q, const T* b, int deg_r, int deg_b, T lc_b_inv)
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

  /*============================== Slice ==============================*/
  template <typename T>
  __global__ void Slice(const T* in, T* out, int offset, int stride, int size)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) { out[tid] = in[offset + tid * stride]; }
  }

} // namespace polynomials