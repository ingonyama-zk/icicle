#pragma once
#include "../../curves/curve_config.cuh"

const uint32_t MAX_NUM_THREADS = 1024;
const uint32_t MAX_THREADS_BATCH = 128;

/**
 * Copy twiddle factors array to device (returns a pointer to the device allocated array).
 * @param twiddles input empty array.
 * @param n_twiddles length of twiddle factors.
 */
scalar_t *copy_twiddle_factors_to_device(scalar_t *twiddles, uint32_t n_twiddles)
{
  size_t size_twiddles = n_twiddles * sizeof(scalar_t);
  scalar_t *d_twiddles;
  cudaMalloc(&d_twiddles, size_twiddles);
  cudaMemcpy(d_twiddles, twiddles, size_twiddles, cudaMemcpyHostToDevice);
  return d_twiddles;
}

/**
 * Computes the twiddle factors.
 * Outputs: d_twiddles[i] = omega^i.
 * @param d_twiddles input empty array.
 * @param n_twiddles number of twiddle factors.
 * @param omega multiplying factor.
 */
__global__ void twiddle_factors_kernel(scalar_t *d_twiddles, uint32_t n_twiddles, scalar_t omega)
{
  for (uint32_t i = 0; i < n_twiddles; i++)
  {
    d_twiddles[i] = scalar_t::zero();
  }
  d_twiddles[0] = scalar_t::one();
  for (uint32_t i = 0; i < n_twiddles - 1; i++)
  {
    d_twiddles[i + 1] = omega * d_twiddles[i];
  }
}

/**
 * Fills twiddles array with twiddle factors.
 * @param twiddles input empty array.
 * @param n_twiddles number of twiddle factors.
 * @param omega multiplying factor.
 */
scalar_t *fill_twiddle_factors_array(uint32_t n_twiddles, scalar_t omega)
{
  size_t size_twiddles = n_twiddles * sizeof(scalar_t);
  scalar_t *d_twiddles;
  cudaMalloc(&d_twiddles, size_twiddles);
  twiddle_factors_kernel<<<1, 1>>>(d_twiddles, n_twiddles, omega);
  return d_twiddles;
}

/**
 * Returens the bit reversed order of a number.
 * for example: on inputs num = 6 (110 in binary) and logn = 3
 * the function should return 3 (011 in binary.)
 * @param num some number with bit representation of size logn.
 * @param logn length of bit representation of num.
 * @return bit reveresed order or num.
 */
__device__ __host__ uint32_t reverseBits(uint32_t num, uint32_t logn)
{
  unsigned int reverse_num = 0;
  int i;
  for (i = 0; i < logn; i++)
  {
    if ((num & (1 << i)))
      reverse_num |= 1 << ((logn - 1) - i);
  }
  return reverse_num;
}

/**
 * Returens the bit reversal ordering of the input array.
 * for example: on input ([a[0],a[1],a[2],a[3]], 4, 2) it returns
 * [a[0],a[3],a[2],a[1]] (elements in indices 3,1 swhich places).
 * @param arr array of some object of type T of size which is a power of 2.
 * @param n length of arr.
 * @param logn log(n).
 * @return A new array which is the bit reversed version of input array.
 */
template <typename T>
T *template_reverse_order(T *arr, uint32_t n, uint32_t logn)
{
  T *arrReversed = new T[n];
  for (uint32_t i = 0; i < n; i++)
  {
    uint32_t reversed = reverseBits(i, logn);
    arrReversed[i] = arr[reversed];
  }
  return arrReversed;
}

/**
 * Cooley-Tuckey butterfly kernel.
 * @param arr array of objects of type E (elements).
 * @param twiddles array of twiddle factors of type S (scalars).
 * @param n size of arr.
 * @param n_twiddles size of omegas.
 * @param m "pair distance" - indicate distance of butterflies inputs.
 * @param i Cooley-TUckey FFT stage number.
 * @param max_thread_num maximal number of threads in stage.
 */
template <typename E, typename S>
__global__ void template_butterfly_kernel(E *arr, S *twiddles, uint32_t n, uint32_t n_twiddles, uint32_t m, uint32_t i, uint32_t max_thread_num)
{
  int j = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (j < max_thread_num)
  {
    uint32_t g = j * (n / m);
    uint32_t k = i + j + (m >> 1);
    E u = arr[i + j];
    E v = twiddles[g * n_twiddles / n] * arr[k];
    arr[i + j] = u + v;
    arr[k] = u - v;
  }
}

/**
 * Set the elements of arr to be the elements of res multiplied by some scalar.
 * @param arr input array.
 * @param res result array.
 * @param n size of arr.
 * @param n_inv scalar of type S (scalar).
 */
template <typename E, typename S>
__global__ void template_normalize_kernel(E *arr, E *res, uint32_t n, S scalar)
{
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < n)
  {
    res[tid] = scalar * arr[tid];
  }
}

/**
 * Cooley-Tuckey NTT.
 * NOTE! this function assumes that d_arr and d_twiddles are located in the device memory.
 * @param d_arr input array of type E (elements) allocated on the device memory.
 * @param n length of d_arr.
 * @param logn log(n).
 * @param d_twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2).
 * @param n_twiddles length of d_twiddles.
 */
template <typename E, typename S>
void template_ntt_on_device_memory(E *d_arr, uint32_t n, uint32_t logn, S *d_twiddles, uint32_t n_twiddles)
{
  uint32_t m = 2;
  for (uint32_t s = 0; s < logn; s++)
  {
    for (uint32_t i = 0; i < n; i += m)
    {
      int shifted_m = m >> 1;
      int number_of_threads = min(shifted_m, MAX_NUM_THREADS);
      int number_of_blocks = (shifted_m + number_of_threads - 1) / number_of_threads;
      template_butterfly_kernel<E, S><<<number_of_blocks, number_of_threads>>>(d_arr, d_twiddles, n, n_twiddles, m, i, shifted_m);
    }
    m <<= 1;
  }
}

/**
 * Cooley-Tuckey NTT.
 * NOTE! this function assumes that d_twiddles are located in the device memory.
 * @param arr input array of type E (elements).
 * @param n length of d_arr.
 * @param d_twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2).
 * @param n_twiddles length of d_twiddles.
 * @param inverse indicate if the result array should be normalized by n^(-1).
 */
template <typename E, typename S>
E *ntt_template(E *arr, uint32_t n, S *d_twiddles, uint32_t n_twiddles, bool inverse)
{
  uint32_t logn = uint32_t(log(n) / log(2));
  size_t size_E = n * sizeof(E);
  E *arrReversed = template_reverse_order<E>(arr, n, logn);
  E *d_arrReversed;
  cudaMalloc(&d_arrReversed, size_E);
  cudaMemcpy(d_arrReversed, arrReversed, size_E, cudaMemcpyHostToDevice);
  template_ntt_on_device_memory<E, S>(d_arrReversed, n, logn, d_twiddles, n_twiddles);
  if (inverse == true)
  {
    int NUM_THREADS = MAX_NUM_THREADS;
    int NUM_BLOCKS = (n + NUM_THREADS - 1) / NUM_THREADS;
    template_normalize_kernel<E, S><<<NUM_THREADS, NUM_BLOCKS>>>(d_arrReversed, d_arrReversed, n, S::inv_log_size(logn));
  }
  cudaMemcpy(arrReversed, d_arrReversed, size_E, cudaMemcpyDeviceToHost);
  cudaFree(d_arrReversed);
  return arrReversed;
}

/**
 * Cooley-Tuckey Elliptic Curve NTT.
 * NOTE! this function assumes that d_twiddles are located in the device memory.
 * @param arr input array of type projective_t.
 * @param n length of d_arr.
 * @param d_twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2).
 * @param n_twiddles length of d_twiddles.
 * @param inverse indicate if the result array should be normalized by n^(-1).
 */
projective_t *ecntt(projective_t *arr, uint32_t n, scalar_t *d_twiddles, uint32_t n_twiddles, bool inverse)
{
  return ntt_template<projective_t, scalar_t>(arr, n, d_twiddles, n_twiddles, inverse);
}

/**
 * Cooley-Tuckey (scalar) NTT.
 * NOTE! this function assumes that d_twiddles are located in the device memory.
 * @param arr input array of type scalar_t.
 * @param n length of d_arr.
 * @param d_twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2).
 * @param n_twiddles length of d_twiddles.
 * @param inverse indicate if the result array should be normalized by n^(-1).
 */
scalar_t *ntt(scalar_t *arr, uint32_t n, scalar_t *d_twiddles, uint32_t n_twiddles, bool inverse)
{
  return ntt_template<scalar_t, scalar_t>(arr, n, d_twiddles, n_twiddles, inverse);
}

/**
 * Cooley-Tuckey (scalar) NTT.
 * @param arr input array of type scalar_t.
 * @param n length of d_arr.
 * @param inverse indicate if the result array should be normalized by n^(-1).
 */
extern "C" uint32_t ntt_end2end(scalar_t *arr, uint32_t n, bool inverse)
{
  uint32_t logn = uint32_t(log(n) / log(2));
  uint32_t n_twiddles = n; // n_twiddles is set to 4096 as scalar_t::omega() is of that order.
  scalar_t *d_twiddles;
  if (inverse)
  {
    d_twiddles = fill_twiddle_factors_array(n_twiddles, scalar_t::omega_inv(logn));
  }
  else
  {
    d_twiddles = fill_twiddle_factors_array(n_twiddles, scalar_t::omega(logn));
  }
  scalar_t *result = ntt_template<scalar_t, scalar_t>(arr, n, d_twiddles, n_twiddles, inverse);
  for (int i = 0; i < n; i++)
  {
    arr[i] = result[i];
  }
  cudaFree(d_twiddles);
  return 0;
}

/**
 * Cooley-Tuckey (scalar) NTT.
 * @param arr input array of type projective_t.
 * @param n length of d_arr.
 * @param inverse indicate if the result array should be normalized by n^(-1).
 */
extern "C" uint32_t ecntt_end2end(projective_t *arr, uint32_t n, bool inverse)
{
  uint32_t logn = uint32_t(log(n) / log(2));
  uint32_t n_twiddles = n;
  scalar_t *twiddles = new scalar_t[n_twiddles];
  scalar_t *d_twiddles;
  if (inverse)
  {
    d_twiddles = fill_twiddle_factors_array(n_twiddles, scalar_t::omega_inv(logn));
  }
  else
  {
    d_twiddles = fill_twiddle_factors_array(n_twiddles, scalar_t::omega(logn));
  }
  projective_t *result = ntt_template<projective_t, scalar_t>(arr, n, d_twiddles, n_twiddles, inverse);
  for (int i = 0; i < n; i++)
  {
    arr[i] = result[i];
  }
  cudaFree(d_twiddles);
  return 0; // TODO add
}

/**
 * Returens the bit reversal ordering of the input array according to the batches *in place*.
 * The assumption is that arr is divided into N tasks of size n.
 * Tasks indicates the index of the task (out of N).
 * @param arr input array of type T.
 * @param n length of arr.
 * @param logn log(n).
 * @param task log(n).
 */
template <typename T>
__device__ __host__ void reverseOrder_batch(T *arr, uint32_t n, uint32_t logn, uint32_t task)
{
  for (uint32_t i = 0; i < n; i++)
  {
    uint32_t reversed = reverseBits(i, logn);
    if (reversed > i)
    {
      T tmp = arr[task * n + i];
      arr[task * n + i] = arr[task * n + reversed];
      arr[task * n + reversed] = tmp;
    }
  }
}

/**
 * Cooley-Tuckey butterfly kernel.
 * @param arr array of objects of type E (elements).
 * @param twiddles array of twiddle factors of type S (scalars).
 * @param n size of arr.
 * @param n_twiddles size of omegas.
 * @param m "pair distance" - indicate distance of butterflies inputs.
 * @param i Cooley-TUckey FFT stage number.
 * @param offset offset corr. to the specific taks (in batch).
 */
template <typename E, typename S>
__device__ __host__ void butterfly(E *arrReversed, S *omegas, uint32_t n, uint32_t n_omegas, uint32_t m, uint32_t i, uint32_t j, uint32_t offset)
{
  uint32_t k = i + j + (m >> 1);
  E u = arrReversed[offset + i + j];
  E v = omegas[j * n_omegas / m] * arrReversed[offset + k];
  arrReversed[offset + i + j] = u + v;
  arrReversed[offset + k] = u - v;
}

/**
 * Cooley-Tuckey NTT.
 * NOTE! this function assumes that d_twiddles are located in the device memory.
 * @param arr input array of type E (elements).
 * @param n length of d_arr.
 * @param d_twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2).
 * @param n_twiddles length of d_twiddles.
 */
template <typename E, typename S>
__global__ void ntt_template_kernel(E *arr, uint32_t n, uint32_t logn, S *twiddles, uint32_t n_twiddles, uint32_t max_task)
{
  int task = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (task < max_task)
  {
    uint32_t offset = task * n;
    reverseOrder_batch<E>(arr, n, logn, task);

    for (uint32_t s = 0; s < logn; s++)
    {
      uint32_t i = 0;
      uint32_t shift_s = 1 << s;
      uint32_t shift2_s = 2 << s;
      uint32_t loop_limit = n >> 1;
      uint32_t n_twiddles_div = n_twiddles >> (s + 1); // Equivalent to: n_twiddles / (2 << s)

      for (uint32_t l = 0; l < loop_limit; l++)
      {
        uint32_t j = l & (shift_s - 1); // Equivalent to: l % (1 << s)
        uint32_t k = i + j + shift_s;

        E u = arr[offset + i + j];
        E v = twiddles[j * n_twiddles_div] * arr[offset + k];
        arr[offset + i + j] = u + v;
        arr[offset + k] = u - v;

        if (j == 0)
        {
          i = (i + shift2_s) % n;
        }
      }
    }
  }
}

/**
 * Cooley-Tuckey (scalar) NTT.
 * This is a bached version - meaning it assumes than the input array
 * consists of N arrays of size n. The function performs n-size NTT on each small array.
 * @param arr input array of type scalar_t.
 * @param arr_size number of total elements = n * N.
 * @param n size of batch.
 * @param inverse indicate if the result array should be normalized by n^(-1).
 */
extern "C" uint32_t ntt_end2end_batch(scalar_t *arr, uint32_t arr_size, uint32_t n, bool inverse)
{
  uint32_t logn = uint32_t(log(n) / log(2));
  uint32_t n_twiddles = n; // n_twiddles is set to 4096 as scalar_t::omega() is of that order.
  size_t size_E = arr_size * sizeof(scalar_t);
  scalar_t *d_twiddles;
  if (inverse)
  {
    d_twiddles = fill_twiddle_factors_array(n_twiddles, scalar_t::omega_inv(logn));
  }
  else
  {
    d_twiddles = fill_twiddle_factors_array(n_twiddles, scalar_t::omega(logn));
  }
  scalar_t *d_arr;
  cudaMalloc(&d_arr, size_E);
  cudaMemcpy(d_arr, arr, size_E, cudaMemcpyHostToDevice);
  int NUM_THREADS = MAX_THREADS_BATCH;
  int NUM_BLOCKS = (int(arr_size / n) + NUM_THREADS - 1) / NUM_THREADS;
  ntt_template_kernel<scalar_t, scalar_t><<<NUM_BLOCKS, NUM_THREADS>>>(d_arr, n, logn, d_twiddles, n_twiddles, int(arr_size / n));
  if (inverse == true)
  {
    NUM_THREADS = MAX_NUM_THREADS;
    NUM_BLOCKS = (arr_size + NUM_THREADS - 1) / NUM_THREADS;
    template_normalize_kernel<scalar_t, scalar_t><<<NUM_BLOCKS, NUM_THREADS>>>(d_arr, d_arr, arr_size, scalar_t::inv_log_size(logn));
  }
  cudaMemcpy(arr, d_arr, size_E, cudaMemcpyDeviceToHost);
  cudaFree(d_arr);
  cudaFree(d_twiddles);
  return 0;
}

/**
 * Cooley-Tuckey (scalar) NTT.
 * This is a bached version - meaning it assumes than the input array
 * consists of N arrays of size n. The function performs n-size NTT on each small array.
 * @param arr input array of type scalar_t.
 * @param arr_size number of total elements = n * N.
 * @param n size of batch.
 * @param inverse indicate if the result array should be normalized by n^(-1).
 */
extern "C" uint32_t ecntt_end2end_batch(projective_t *arr, uint32_t arr_size, uint32_t n, bool inverse)
{
  uint32_t logn = uint32_t(log(n) / log(2));
  uint32_t n_twiddles = n; // n_twiddles is set to 4096 as scalar_t::omega() is of that order.
  size_t size_E = arr_size * sizeof(projective_t);
  scalar_t *d_twiddles;
  if (inverse)
  {
    d_twiddles = fill_twiddle_factors_array(n_twiddles, scalar_t::omega_inv(logn));
  }
  else
  {
    d_twiddles = fill_twiddle_factors_array(n_twiddles, scalar_t::omega(logn));
  }
  projective_t *d_arr;
  cudaMalloc(&d_arr, size_E);
  cudaMemcpy(d_arr, arr, size_E, cudaMemcpyHostToDevice);
  int NUM_THREADS = MAX_THREADS_BATCH;
  int NUM_BLOCKS = (int(arr_size / n) + NUM_THREADS - 1) / NUM_THREADS;
  ntt_template_kernel<projective_t, scalar_t><<<NUM_BLOCKS, NUM_THREADS>>>(d_arr, n, logn, d_twiddles, n_twiddles, int(arr_size / n));
  if (inverse == true)
  {
    NUM_THREADS = MAX_NUM_THREADS;
    NUM_BLOCKS = (arr_size + NUM_THREADS - 1) / NUM_THREADS;
    //TODO: no correctnes when swapped
    template_normalize_kernel<projective_t, scalar_t><<<NUM_THREADS, NUM_BLOCKS>>>(d_arr, d_arr, arr_size, scalar_t::inv_log_size(logn));
  }
  cudaMemcpy(arr, d_arr, size_E, cudaMemcpyDeviceToHost);
  cudaFree(d_arr);
  cudaFree(d_twiddles);
  return 0;
}