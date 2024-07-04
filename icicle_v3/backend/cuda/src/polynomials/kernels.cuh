#pragma once
#include "stdint.h"
// #include "../../../src/vec_ops/vec_ops.cu" // TODO Yuval: avoid this

namespace polynomials {
  // TODO Yuval remove vec_ops kernels from here

  template <typename E>
  __global__ void mul_kernel(const E* scalar_vec, const E* element_vec, int n, E* result)
  {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) { result[tid] = scalar_vec[tid] * element_vec[tid]; }
  }

  template <typename E, typename S>
  __global__ void mul_scalar_kernel(const E* element_vec, const S scalar, int n, E* result)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) { result[tid] = element_vec[tid] * (scalar); }
  }

  template <typename E>
  __global__ void div_element_wise_kernel(const E* element_vec1, const E* element_vec2, int n, E* result)
  {
    // TODO:implement better based on https://eprint.iacr.org/2008/199
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) { result[tid] = element_vec1[tid] * E::inverse(element_vec2[tid]); }
  }

  /*============================== add/sub ==============================*/
  template <typename T>
  __global__ void add_sub_kernel(const T* a_vec, const T* b_vec, int a_len, int b_len, bool add1_sub0, T* result)
  {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= max(a_len, b_len)) return;

    T a = tid >= a_len ? T::zero() : a_vec[tid];
    T b = tid >= b_len ? T::zero() : b_vec[tid];
    result[tid] = add1_sub0 ? a + b : a - b;
  }

  // Note: must be called with 1 block, 1 thread
  template <typename T>
  __global__ void add_single_element_inplace(T* self, T v)
  {
    *self = *self + v;
  }

  /*============================== degree ==============================*/
  template <typename T>
  __global__ void highest_non_zero_idx(const T* vec, int len, int64_t* idx)
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
  // TODO Yuval: implement efficient reduction and support batch evaluation
  template <typename T>
  __global__ void dummy_reduce(const T* arr, int size, T* output)
  {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0) return;

    *output = arr[0];
    for (int i = 1; i < size; ++i) {
      *output = *output + arr[i];
    }
  }

  template <typename T>
  __global__ void evaluate_polynomial_without_reduction(const T* x, const T* coeffs, int num_coeffs, T* tmp)
  {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_coeffs) { tmp[tid] = coeffs[tid] * T::pow(*x, tid); }
  }

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

  /*============================== slice ==============================*/
  template <typename T>
  __global__ void slice_kernel(const T* in, T* out, int offset, int stride, int size)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) { out[tid] = in[offset + tid * stride]; }
  }

} // namespace polynomials