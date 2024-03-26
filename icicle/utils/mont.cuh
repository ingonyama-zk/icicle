#pragma once
#ifndef MONT_H
#define MONT_H

namespace mont {

  namespace {

#define MAX_THREADS_PER_BLOCK 256

    template <typename E, bool is_into>
    __global__ void MontgomeryKernel(const E* input, int n, E* output)
    {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < n) { output[tid] = is_into ? E::ToMontgomery(input[tid]) : E::FromMontgomery(input[tid]); }
    }

    template <typename E, bool is_into>
    cudaError_t ConvertMontgomery(const E* d_input, int n, cudaStream_t stream, E* d_output)
    {
      // Set the grid and block dimensions
      int num_threads = MAX_THREADS_PER_BLOCK;
      int num_blocks = (n + num_threads - 1) / num_threads;
      MontgomeryKernel<E, is_into><<<num_blocks, num_threads, 0, stream>>>(d_input, n, d_output);

      return CHK_LAST();
    }

  } // namespace

  template <typename E>
  cudaError_t ToMontgomery(const E* d_input, int n, cudaStream_t stream, E* d_output)
  {
    return ConvertMontgomery<E, true>(d_input, n, stream, d_output);
  }

  template <typename E>
  cudaError_t FromMontgomery(const E* d_input, int n, cudaStream_t stream, E* d_output)
  {
    return ConvertMontgomery<E, false>(d_input, n, stream, d_output);
  }

} // namespace mont

#endif
