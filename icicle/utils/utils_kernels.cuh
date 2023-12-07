#pragma once
#ifndef UTILS_KERNELS_H
#define UTILS_KERNELS_H

namespace utils_internal {

  template <typename E, typename S>
  __global__ void NormalizeKernel(E* arr, S scalar, int n);

  template <typename E, typename S>
  __global__ void BatchMulKernel(E* in_vec, S* scalar_vec, int n_scalars, int batch_size, E* out_vec);

} // namespace utils_internal

#endif
