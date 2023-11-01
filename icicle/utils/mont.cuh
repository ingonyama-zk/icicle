#pragma once
#ifndef MONT_H
#define MONT_H

namespace mont {

  namespace {

#define MAX_THREADS_PER_BLOCK 256

template <typename E, bool is_into>
__global__ void MontgomeryKernel(E* inout, int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) { inout[tid] = is_into ? E::ToMontgomery(inout[tid]) : E::FromMontgomery(inout[tid]); }
}

template <typename E, bool is_into>
int ConvertMontgomery(E* d_inout, int n, cudaStream_t stream)
{
  // Set the grid and block dimensions
  int num_threads = MAX_THREADS_PER_BLOCK;
  int num_blocks = (n + num_threads - 1) / num_threads;
  MontgomeryKernel<E, is_into><<<num_blocks, num_threads, 0, stream>>>(d_inout, n);

      return 0; // TODO: void with propper error handling
    }

  } // namespace

template <typename E>
int ToMontgomery(E* d_inout, int n, cudaStream_t stream)
{
  return ConvertMontgomery<E, true>(d_inout, n, stream);
}

template <typename E>
int FromMontgomery(E* d_inout, int n, cudaStream_t stream)
{
  return ConvertMontgomery<E, false>(d_inout, n, stream);
}

} // namespace mont

#endif
