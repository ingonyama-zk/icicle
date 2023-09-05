#pragma once

#include "../appUtils/vector_manipulation/ve_mod_mult.cuh"

template <typename E>
int convert_montgomery(E* d_inout, size_t n_elments, bool is_into, cudaStream_t stream)
{
  // Set the grid and block dimensions
  int num_threads = MAX_THREADS_PER_BLOCK;
  int num_blocks = (n_elments + num_threads - 1) / num_threads;
  E mont = is_into ? E::montgomery_r() : E::montgomery_r_inv();
  template_normalize_kernel<<<num_blocks, num_threads, 0, stream>>>(d_inout, n_elments, mont);

  return 0; // TODO: void with propper error handling
}

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