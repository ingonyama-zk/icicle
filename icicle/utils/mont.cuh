#pragma once

#define MAX_THREADS_PER_BLOCK 256

template <typename E, bool is_into>
__global__ void montgomery_kernel(E* inout, int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) { inout[tid] = is_into ? E::to_montgomery(inout[tid]) : E::from_montgomery(inout[tid]); }
}

template <typename E, bool is_into>
int convert_montgomery(E* d_inout, int n, cudaStream_t stream)
{
  // Set the grid and block dimensions
  int num_threads = MAX_THREADS_PER_BLOCK;
  int num_blocks = (n + num_threads - 1) / num_threads;
  montgomery_kernel<E, is_into><<<num_blocks, num_threads, 0, stream>>>(d_inout, n);

  return 0; // TODO: void with propper error handling
}

template <typename E>
int to_montgomery(E* d_inout, unsigned n, cudaStream_t stream)
{
  return convert_montgomery<E, true>(d_inout, n, stream);
}

template <typename E>
int from_montgomery(E* d_inout, unsigned n, cudaStream_t stream)
{
  return convert_montgomery<E, false>(d_inout, n, stream);
}