#pragma once
#ifndef UTILS_KERNELS_H
#define UTILS_KERNELS_H

namespace utils_internal {
  template <typename E, typename S>
  void NormalizeKernel(E* arr, S scalar, int n)
  {
    return;
  }

  template <typename E, typename S>
  void BatchMulKernel(
    const E* in_vec,
    int n_elements,
    int batch_size,
    S* scalar_vec,
    int step,
    int n_scalars,
    int logn,
    bool bitrev,
    E* out_vec)
  {
    return;
  }

} // namespace utils_internal

#endif