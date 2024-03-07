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
  __global__ void HighestNonZeroIdx(T* vec, int len, int64_t* idx)
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
  __global__ void evaluatePolynomialWithoutReduction(T x, T* coeffs, int num_coeffs, T* tmp)
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

  template <typename T>
  __global__ void MulScalar(T* element_vec, T* scalar, int n, T* result)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) { result[tid] = element_vec[tid] * (*scalar); }
  }

  /*============================== division ==============================*/
  template <typename T>
  __global__ void SchoolBookDivisionStep(T* r, T* q, T* b, int deg_r, int deg_b, T lc_b_inv)
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

  /*============================== division_by_vanishing_polynomial ==============================*/
  template <typename T>
  __global__ void DivElementWise(T* element_vec1, T* element_vec2, int n, T* result)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) { result[tid] = element_vec1[tid] * T::inverse(element_vec2[tid]); }
  }

} // namespace polynomials