#ifndef NTT
#define NTT
#pragma once

#include "../../utils/sharedmem.cuh"
#include "../vector_manipulation/ve_mod_mult.cuh"

const uint32_t MAX_NUM_THREADS = 512;
const uint32_t MAX_THREADS_BATCH = 512;          // TODO: allows 100% occupancy for scalar NTT for sm_86..sm_89
const uint32_t MAX_SHARED_MEM_ELEMENT_SIZE = 32; // TODO: occupancy calculator, hardcoded for sm_86..sm_89
const uint32_t MAX_SHARED_MEM = MAX_SHARED_MEM_ELEMENT_SIZE * 1024;

/**
 * Computes the twiddle factors.
 * Outputs: d_twiddles[i] = omega^i.
 * @param d_twiddles input empty array.
 * @param n_twiddles number of twiddle factors.
 * @param omega multiplying factor.
 */
template <typename S>
__global__ void twiddle_factors_kernel(S* d_twiddles, uint32_t n_twiddles, S omega)
{
  for (uint32_t i = 0; i < n_twiddles; i++) {
    d_twiddles[i] = S::zero();
  }
  d_twiddles[0] = S::one();
  for (uint32_t i = 0; i < n_twiddles - 1; i++) {
    d_twiddles[i + 1] = omega * d_twiddles[i];
  }
}

/**
 * Fills twiddles array with twiddle factors.
 * @param twiddles input empty array.
 * @param n_twiddles number of twiddle factors.
 * @param omega multiplying factor.
 */
template <typename S>
S* fill_twiddle_factors_array(uint32_t n_twiddles, S omega, cudaStream_t stream)
{
  size_t size_twiddles = n_twiddles * sizeof(S);
  S* d_twiddles;
  cudaMallocAsync(&d_twiddles, size_twiddles, stream);
  twiddle_factors_kernel<S><<<1, 1, 0, stream>>>(d_twiddles, n_twiddles, omega);
  cudaStreamSynchronize(stream);
  return d_twiddles;
}

template <typename T>
__global__ void reverse_order_kernel(T* arr, T* arr_reversed, uint32_t n, uint32_t logn, uint32_t batch_size)
{
  int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (threadId < n * batch_size) {
    int idx = threadId % n;
    int batch_idx = threadId / n;
    int idx_reversed = __brev(idx) >> (32 - logn);
    arr_reversed[batch_idx * n + idx_reversed] = arr[batch_idx * n + idx];
  }
}

/**
 * Bit-reverses a batch of input arrays in-place inside GPU.
 * for example: on input array ([a[0],a[1],a[2],a[3]], 4, 2) it returns
 * [a[0],a[3],a[2],a[1]] (elements at indices 3 and 1 swhich places).
 * @param arr batch of arrays of some object of type T. Should be on GPU.
 * @param n length of `arr`.
 * @param logn log(n).
 * @param batch_size the size of the batch.
 */
template <typename T>
void reverse_order_batch(T* arr, uint32_t n, uint32_t logn, uint32_t batch_size, cudaStream_t stream)
{
  T* arr_reversed;
  cudaMallocAsync(&arr_reversed, n * batch_size * sizeof(T), stream);
  int number_of_threads = MAX_THREADS_BATCH;
  int number_of_blocks = (n * batch_size + number_of_threads - 1) / number_of_threads;
  reverse_order_kernel<<<number_of_blocks, number_of_threads, 0, stream>>>(arr, arr_reversed, n, logn, batch_size);
  cudaMemcpyAsync(arr, arr_reversed, n * batch_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
  cudaFreeAsync(arr_reversed, stream);
}

/**
 * Bit-reverses an input array in-place inside GPU.
 * for example: on array ([a[0],a[1],a[2],a[3]], 4, 2) it returns
 * [a[0],a[3],a[2],a[1]] (elements at indices 3 and 1 swhich places).
 * @param arr array of some object of type T of size which is a power of 2. Should be on GPU.
 * @param n length of `arr`.
 * @param logn log(n).
 */
template <typename T>
void reverse_order(T* arr, uint32_t n, uint32_t logn, cudaStream_t stream)
{
  reverse_order_batch(arr, n, logn, 1, stream);
}

/**
 * Cooley-Tuckey NTT.
 * NOTE! this function assumes that d_twiddles are located in the device memory.
 * @param arr input array of type E (elements).
 * @param n length of d_arr.
 * @param twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2).
 * @param n_twiddles length of twiddles.
 * @param max_task max count of parallel tasks.
 * @param s log2(n) loop index.
 */
template <typename E, typename S>
__global__ void ntt_template_kernel_shared_rev(
  E* __restrict__ arr_g,
  uint32_t n,
  const S* __restrict__ r_twiddles,
  uint32_t n_twiddles,
  uint32_t max_task,
  uint32_t ss,
  uint32_t logn)
{
  SharedMemory<E> smem;
  E* arr = smem.getPointer();

  uint32_t task = blockIdx.x;
  uint32_t loop_limit = blockDim.x;
  uint32_t chunks = n / (loop_limit * 2);
  uint32_t offset = (task / chunks) * n;
  if (task < max_task) {
    // flattened loop allows parallel processing
    uint32_t l = threadIdx.x;

    if (l < loop_limit) {
#pragma unroll
      for (; ss < logn; ss++) {
        int s = logn - ss - 1;
        bool is_beginning = ss == 0;
        bool is_end = ss == (logn - 1);

        uint32_t ntw_i = task % chunks;

        uint32_t n_twiddles_div = n_twiddles >> (s + 1);

        uint32_t shift_s = 1 << s;
        uint32_t shift2_s = 1 << (s + 1);

        l = ntw_i * loop_limit + l; // to l from chunks to full

        uint32_t j = l & (shift_s - 1);               // Equivalent to: l % (1 << s)
        uint32_t i = ((l >> s) * shift2_s) & (n - 1); // (..) % n (assuming n is power of 2)
        uint32_t oij = i + j;
        uint32_t k = oij + shift_s;

        S tw = r_twiddles[j * n_twiddles_div];

        E u = is_beginning ? arr_g[offset + oij] : arr[oij];
        E v = is_beginning ? arr_g[offset + k] : arr[k];
        if (is_end) {
          arr_g[offset + oij] = u + v;
          arr_g[offset + k] = tw * (u - v);
        } else {
          arr[oij] = u + v;
          arr[k] = tw * (u - v);
        }

        __syncthreads();
      }
    }
  }
}

/**
 * Cooley-Tuckey NTT.
 * NOTE! this function assumes that d_twiddles are located in the device memory.
 * @param arr input array of type E (elements).
 * @param n length of d_arr.
 * @param twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2).
 * @param n_twiddles length of twiddles.
 * @param max_task max count of parallel tasks.
 * @param s log2(n) loop index.
 */
template <typename E, typename S>
__global__ void ntt_template_kernel_shared(
  E* __restrict__ arr_g,
  uint32_t n,
  const S* __restrict__ r_twiddles,
  uint32_t n_twiddles,
  uint32_t max_task,
  uint32_t s,
  uint32_t logn)
{
  SharedMemory<E> smem;
  E* arr = smem.getPointer();

  uint32_t task = blockIdx.x;
  uint32_t loop_limit = blockDim.x;
  uint32_t chunks = n / (loop_limit * 2);
  uint32_t offset = (task / chunks) * n;
  if (task < max_task) {
    // flattened loop allows parallel processing
    uint32_t l = threadIdx.x;

    if (l < loop_limit) {
#pragma unroll
      for (; s < logn; s++) // TODO: this loop also can be unrolled
      {
        uint32_t ntw_i = task % chunks;

        uint32_t n_twiddles_div = n_twiddles >> (s + 1);

        uint32_t shift_s = 1 << s;
        uint32_t shift2_s = 1 << (s + 1);

        l = ntw_i * loop_limit + l; // to l from chunks to full

        uint32_t j = l & (shift_s - 1);               // Equivalent to: l % (1 << s)
        uint32_t i = ((l >> s) * shift2_s) & (n - 1); // (..) % n (assuming n is power of 2)
        uint32_t oij = i + j;
        uint32_t k = oij + shift_s;
        S tw = r_twiddles[j * n_twiddles_div];

        E u = s == 0 ? arr_g[offset + oij] : arr[oij];
        E v = s == 0 ? arr_g[offset + k] : arr[k];
        v = tw * v;
        if (s == (logn - 1)) {
          arr_g[offset + oij] = u + v;
          arr_g[offset + k] = u - v;
        } else {
          arr[oij] = u + v;
          arr[k] = u - v;
        }

        __syncthreads();
      }
    }
  }
}

/**
 * Cooley-Tukey NTT.
 * NOTE! this function assumes that d_twiddles are located in the device memory.
 * @param arr input array of type E (elements).
 * @param n length of d_arr.
 * @param twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2).
 * @param n_twiddles length of twiddles.
 * @param max_task max count of parallel tasks.
 * @param s log2(n) loop index.
 */
template <typename E, typename S>
__global__ void
ntt_template_kernel(E* arr, uint32_t n, S* twiddles, uint32_t n_twiddles, uint32_t max_task, uint32_t s, bool rev)
{
  int task = blockIdx.x;
  int chunks = n / (blockDim.x * 2);

  if (task < max_task) {
    // flattened loop allows parallel processing
    uint32_t l = threadIdx.x;
    uint32_t loop_limit = blockDim.x;

    if (l < loop_limit) {
      uint32_t ntw_i = task % chunks;

      uint32_t shift_s = 1 << s;
      uint32_t shift2_s = 1 << (s + 1);
      uint32_t n_twiddles_div = n_twiddles >> (s + 1);

      l = ntw_i * blockDim.x + l; // to l from chunks to full

      uint32_t j = l & (shift_s - 1);               // Equivalent to: l % (1 << s)
      uint32_t i = ((l >> s) * shift2_s) & (n - 1); // (..) % n (assuming n is power of 2)
      uint32_t k = i + j + shift_s;

      S tw = twiddles[j * n_twiddles_div];

      uint32_t offset = (task / chunks) * n;
      E u = arr[offset + i + j];
      E v = arr[offset + k];
      if (!rev) v = tw * v;
      arr[offset + i + j] = u + v;
      v = u - v;
      arr[offset + k] = rev ? tw * v : v;
    }
  }
}

/**
 * NTT/INTT inplace batch
 * Note: this function does not preform any bit-reverse permutations on its inputs or outputs.
 * @param d_inout Array for inplace processing
 * @param d_twiddles
 * @param n Length of `d_twiddles` array
 * @param batch_size The size of the batch; the length of `d_inout` is `n` * `batch_size`.
 * @param inverse true for iNTT
 * @param is_coset true for multiplication by coset
 * @param coset should be array of lenght n - or in case of lesser than n, right-padded with zeroes
 * @param stream CUDA stream
 * @param is_sync_needed do perform sync of the supplied CUDA stream at the end of processing
 */
template <typename E, typename S>
void ntt_inplace_batch_template(
  E* d_inout,
  S* d_twiddles,
  unsigned n,
  unsigned batch_size,
  bool inverse,
  bool is_coset,
  S* coset,
  cudaStream_t stream,
  bool is_sync_needed)
{
  const int logn = int(log(n) / log(2));
  bool is_shared_mem_enabled = sizeof(E) <= MAX_SHARED_MEM_ELEMENT_SIZE;
  const int log2_shmem_elems = is_shared_mem_enabled ? int(log(int(MAX_SHARED_MEM / sizeof(E))) / log(2)) : logn;
  int num_threads = min(min(n / 2, MAX_THREADS_BATCH), 1 << (log2_shmem_elems - 1));
  const int chunks = max(int((n / 2) / num_threads), 1);
  const int total_tasks = batch_size * chunks;
  int num_blocks = total_tasks;
  const int shared_mem = 2 * num_threads * sizeof(E); // TODO: calculator, as shared mem size may be more efficient less
                                                      // then max to allow more concurrent blocks on SM
  const int logn_shmem = is_shared_mem_enabled ? int(log(2 * num_threads) / log(2))
                                               : 0; // TODO: shared memory support only for types <= 32 bytes

  if (inverse) {
    if (is_shared_mem_enabled)
      ntt_template_kernel_shared<<<num_blocks, num_threads, shared_mem, stream>>>(
        d_inout, 1 << logn_shmem, d_twiddles, n, total_tasks, 0, logn_shmem);

    for (int s = logn_shmem; s < logn; s++) // TODO: this loop also can be unrolled
    {
      ntt_template_kernel<E, S>
        <<<num_blocks, num_threads, 0, stream>>>(d_inout, n, d_twiddles, n, total_tasks, s, false);
    }

    if (is_coset) batch_vector_mult(coset, d_inout, n, batch_size, stream);

    num_threads = min(n / 2, MAX_NUM_THREADS);
    num_blocks = (n * batch_size + num_threads - 1) / num_threads;
    template_normalize_kernel<E, S>
      <<<num_blocks, num_threads, 0, stream>>>(d_inout, n * batch_size, S::inv_log_size(logn));
  } else {
    if (is_coset) batch_vector_mult(coset, d_inout, n, batch_size, stream);

    for (int s = logn - 1; s >= logn_shmem; s--) // TODO: this loop also can be unrolled
    {
      ntt_template_kernel<<<num_blocks, num_threads, 0, stream>>>(d_inout, n, d_twiddles, n, total_tasks, s, true);
    }

    if (is_shared_mem_enabled)
      ntt_template_kernel_shared_rev<<<num_blocks, num_threads, shared_mem, stream>>>(
        d_inout, 1 << logn_shmem, d_twiddles, n, total_tasks, 0, logn_shmem);
  }

  if (!is_sync_needed) return;

  cudaStreamSynchronize(stream);
}

/**
 * Cooley-Tukey (scalar) NTT.
 * This is a bached version - meaning it assumes than the input array
 * consists of N arrays of size n. The function performs n-size NTT on each small array.
 * @param arr input array of type BLS12_381::scalar_t.
 * @param arr_size number of total elements = n * N.
 * @param n size of batch.
 * @param inverse indicate if the result array should be normalized by n^(-1).
 */
template <typename E, typename S>
uint32_t ntt_end2end_batch_template(E* arr, uint32_t arr_size, uint32_t n, bool inverse, cudaStream_t stream)
{
  int batches = int(arr_size / n);
  uint32_t logn = uint32_t(log(n) / log(2));
  uint32_t n_twiddles = n; // n_twiddles is set to 4096 as BLS12_381::scalar_t::omega() is of that order.
  size_t size_E = arr_size * sizeof(E);
  S* d_twiddles;
  if (inverse) {
    d_twiddles = fill_twiddle_factors_array(n_twiddles, S::omega_inv(logn), stream);
  } else {
    d_twiddles = fill_twiddle_factors_array(n_twiddles, S::omega(logn), stream);
  }
  E* d_arr;
  cudaMallocAsync(&d_arr, size_E, stream);
  cudaMemcpyAsync(d_arr, arr, size_E, cudaMemcpyHostToDevice, stream);
  int NUM_THREADS = MAX_THREADS_BATCH;
  int NUM_BLOCKS = (batches + NUM_THREADS - 1) / NUM_THREADS;

  S* _null = nullptr;
  ntt_inplace_batch_template(d_arr, d_twiddles, n, batches, inverse, false, _null, stream, false);

  cudaMemcpyAsync(arr, d_arr, size_E, cudaMemcpyDeviceToHost, stream);
  cudaFreeAsync(d_arr, stream);
  cudaFreeAsync(d_twiddles, stream);
  cudaStreamSynchronize(stream);
  return 0;
}

/**
 * Cooley-Tukey (scalar) NTT.
 * @param arr input array of type E (element).
 * @param n length of d_arr.
 * @param inverse indicate if the result array should be normalized by n^(-1).
 */
template <typename E, typename S>
uint32_t ntt_end2end_template(E* arr, uint32_t n, bool inverse, cudaStream_t stream)
{
  return ntt_end2end_batch_template<E, S>(arr, n, n, inverse, stream);
}

#endif