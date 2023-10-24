#pragma once
#ifndef MONT_H
#define MONT_H

#include "utils_kernels.cuh"

namespace mont {

  namespace {

#define MAX_THREADS_PER_BLOCK 256

    // TODO (DmytroTym): do valid conversion for point types too
    template <typename E>
    int convert_montgomery(E* d_inout, size_t n, bool is_into, cudaStream_t stream)
    {
      // Set the grid and block dimensions
      int num_threads = MAX_THREADS_PER_BLOCK;
      int num_blocks = (n + num_threads - 1) / num_threads;
      E mont = is_into ? E::montgomery_r() : E::montgomery_r_inv();
      utils_internal::template_normalize_kernel<<<num_blocks, num_threads, 0, stream>>>(d_inout, mont, n);

      return 0; // TODO: void with propper error handling
    }

  } // namespace

  template <typename E>
  int to_montgomery(E* d_inout, unsigned n, cudaStream_t stream)
  {
    return convert_montgomery(d_inout, n, true, stream);
  }

  template <typename E>
  int from_montgomery(E* d_inout, unsigned n, cudaStream_t stream)
  {
    return convert_montgomery(d_inout, n, false, stream);
  }

} // namespace mont

#endif
