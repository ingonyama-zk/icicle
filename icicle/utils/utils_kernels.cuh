#pragma once
#ifndef UTILS_KERNELS_H
#define UTILS_KERNELS_H

#include "utils_kernels.cuh"

namespace utils_internal {
  // TODO: weird linking issue - only works in headers
  // template <typename E, typename S>
  // __global__ void NormalizeKernel(E* arr, S scalar, unsigned n)
  // {
  //   int tid = blockIdx.x * blockDim.x + threadIdx.x;
  //   if (tid < n) { arr[tid] = scalar * arr[tid]; }
  // }

  template <typename E, typename S>
  __global__ void NormalizeKernel(E* arr, S scalar, int n)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) { arr[tid] = scalar * arr[tid]; }
  }

  template <typename E, typename S>
  __global__ void BatchMulKernel(
    E* in_vec,
    int n_elements,
    int batch_size,
    S* scalar_vec,
    int step,
    int n_scalars,
    int logn,
    bool bitrev,
    E* out_vec)
  {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n_elements * batch_size) {
      int64_t scalar_id = tid % n_elements;
      if (bitrev) scalar_id = __brev(scalar_id) >> (32 - logn);
      out_vec[tid] = *(scalar_vec + ((scalar_id * step) % n_scalars)) * in_vec[tid];
    }
  }

} // namespace utils_internal

#endif