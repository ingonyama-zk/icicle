#ifndef NTT
#define NTT
#pragma once

const uint32_t MAX_NUM_THREADS = 1024;
const uint32_t MAX_THREADS_BATCH = 256;

/**
 * Computes the twiddle factors.  
 * Outputs: d_twiddles[i] = omega^i.
 * @param d_twiddles input empty array. 
 * @param n_twiddles number of twiddle factors. 
 * @param omega multiplying factor. 
 */
 template < typename S > __global__ void twiddle_factors_kernel(S * d_twiddles, uint32_t n_twiddles, S omega) {
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
 template < typename S > S * fill_twiddle_factors_array(uint32_t n_twiddles, S omega, cudaStream_t stream) {
  size_t size_twiddles = n_twiddles * sizeof(S);
  S * d_twiddles;
  cudaMallocAsync(& d_twiddles, size_twiddles, stream);
  twiddle_factors_kernel<S> <<< 1, 1, 0, stream>>> (d_twiddles, n_twiddles, omega);
  cudaStreamSynchronize(stream);
  return d_twiddles;
}

/**
 * Returns the bit reversed order of a number. 
 * for example: on inputs num = 6 (110 in binary) and logn = 3
 * the function should return 3 (011 in binary.)
 * @param num some number with bit representation of size logn.
 * @param logn length of bit representation of `num`.
 * @return bit reveresed order or `num`.
 */
__device__ __host__ uint32_t reverseBits(uint32_t num, uint32_t logn) {
  unsigned int reverse_num = 0;
  for (uint32_t i = 0; i < logn; i++) {
    if ((num & (1 << i))) reverse_num |= 1 << ((logn - 1) - i);
  }
  return reverse_num;
}

/**
 * Returns the bit reversal ordering of the input array.
 * for example: on input ([a[0],a[1],a[2],a[3]], 4, 2) it returns
 * [a[0],a[3],a[2],a[1]] (elements in indices 3,1 swhich places).
 * @param arr array of some object of type T of size which is a power of 2. 
 * @param n length of `arr`.
 * @param logn log(n).
 * @return A new array which is the bit reversed version of input array. 
 */
template < typename T > T * template_reverse_order(T * arr, uint32_t n, uint32_t logn) {
  T * arrReversed = new T[n];
  for (uint32_t i = 0; i < n; i++) {
    uint32_t reversed = reverseBits(i, logn);
    arrReversed[i] = arr[reversed];
  }
  return arrReversed;
}

template < typename T > __global__ void reverse_order_kernel(T* arr, T* arr_reversed, uint32_t n, uint32_t logn, uint32_t batch_size) {
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
template < typename T > void reverse_order_batch(T* arr, uint32_t n, uint32_t logn, uint32_t batch_size, cudaStream_t stream) {
  T* arr_reversed;
  cudaMallocAsync(&arr_reversed, n * batch_size * sizeof(T), stream);
  int number_of_threads = MAX_THREADS_BATCH;
  int number_of_blocks = (n * batch_size + number_of_threads - 1) / number_of_threads;
  reverse_order_kernel <<<number_of_blocks, number_of_threads, 0, stream>>> (arr, arr_reversed, n, logn, batch_size);
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
template < typename T > void reverse_order(T* arr, uint32_t n, uint32_t logn, cudaStream_t stream) {
  reverse_order_batch(arr, n, logn, 1, stream);
}

/**
 * Cooley-Tukey butterfly kernel. 
 * @param arr array of objects of type E (elements). 
 * @param twiddles array of twiddle factors of type S (scalars). 
 * @param n size of arr. 
 * @param n_twiddles size of omegas.
 * @param m "pair distance" - indicate distance of butterflies inputs.
 * @param i Cooley-Tukey FFT stage number.
 * @param max_thread_num maximal number of threads in stage. 
 */
template < typename E, typename S > __global__ void template_butterfly_kernel(E * arr, S * twiddles, uint32_t n, uint32_t n_twiddles, uint32_t m, uint32_t i, uint32_t max_thread_num) {
  int j = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (j < max_thread_num) {
    uint32_t g = j * (n / m);
    uint32_t k = i + j + (m >> 1);
    E u = arr[i + j];
    E v = twiddles[g * n_twiddles / n] * arr[k];
    arr[i + j] = u + v;
    arr[k] = u - v;
  }
}

/**
 * Multiply the elements of an input array by a scalar in-place.
 * @param arr input array.
 * @param n size of arr.
 * @param n_inv scalar of type S (scalar).
 */
template < typename E, typename S > __global__ void template_normalize_kernel(E * arr, uint32_t n, S scalar) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < n) {
    arr[tid] = scalar * arr[tid];
  }
}

/**
 * Cooley-Tukey NTT.
 * NOTE! this function assumes that d_arr and d_twiddles are located in the device memory.
 * @param d_arr input array of type E (elements) allocated on the device memory.
 * @param n length of d_arr.
 * @param logn log(n).
 * @param d_twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2).
 * @param n_twiddles length of d_twiddles.
 */
template < typename E, typename S > void template_ntt_on_device_memory(E * d_arr, uint32_t n, uint32_t logn, S * d_twiddles, uint32_t n_twiddles, cudaStream_t stream) {
  uint32_t m = 2;
  // TODO: optimize with separate streams for each iteration
  for (uint32_t s = 0; s < logn; s++) {
    for (uint32_t i = 0; i < n; i += m) {
        uint32_t shifted_m = m >> 1;
        uint32_t number_of_threads = MAX_NUM_THREADS ^ ((shifted_m ^ MAX_NUM_THREADS) & -(shifted_m < MAX_NUM_THREADS));
        uint32_t number_of_blocks = shifted_m / MAX_NUM_THREADS + 1;
        template_butterfly_kernel < E, S > <<< number_of_threads, number_of_blocks, 0, stream >>> (d_arr, d_twiddles, n, n_twiddles, m, i, m >> 1);
    }
    m <<= 1;
  }
}

/**
 * Cooley-Tukey NTT. 
 * NOTE! this function assumes that d_twiddles are located in the device memory.
 * @param arr input array of type E (elements). 
 * @param n length of d_arr.
 * @param d_twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2). 
 * @param n_twiddles length of d_twiddles. 
 * @param inverse indicate if the result array should be normalized by n^(-1). 
 */
template < typename E, typename S > E * ntt_template(E * arr, uint32_t n, S * d_twiddles, uint32_t n_twiddles, bool inverse, cudaStream_t stream) {
  uint32_t logn = uint32_t(log(n) / log(2));
  size_t size_E = n * sizeof(E);
  E * arrReversed = template_reverse_order < E > (arr, n, logn);
  E * d_arrReversed;
  cudaMallocAsync( & d_arrReversed, size_E, stream);
  cudaMemcpyAsync(d_arrReversed, arrReversed, size_E, cudaMemcpyHostToDevice, stream);
  template_ntt_on_device_memory < E, S > (d_arrReversed, n, logn, d_twiddles, n_twiddles, stream);
  if (inverse) {
    int NUM_THREADS = MAX_NUM_THREADS;
    int NUM_BLOCKS = (n + NUM_THREADS - 1) / NUM_THREADS;
    template_normalize_kernel < E, S > <<< NUM_THREADS, NUM_BLOCKS, 0, stream >>> (d_arrReversed, n, S::inv_log_size(logn));
  }
  cudaMemcpyAsync(arrReversed, d_arrReversed, size_E, cudaMemcpyDeviceToHost, stream);
  cudaFreeAsync(d_arrReversed, stream);
  cudaStreamSynchronize(stream);
  return arrReversed;
}

/**
 * Cooley-Tukey (scalar) NTT. 
 * @param arr input array of type E (element). 
 * @param n length of d_arr.
 * @param inverse indicate if the result array should be normalized by n^(-1). 
 */
 template<typename E,typename S> uint32_t ntt_end2end_template(E * arr, uint32_t n, bool inverse, cudaStream_t stream) {
  uint32_t logn = uint32_t(log(n) / log(2));
  uint32_t n_twiddles = n; 
  S * twiddles = new S[n_twiddles];
  S * d_twiddles;
  if (inverse){
    d_twiddles = fill_twiddle_factors_array(n_twiddles, S::omega_inv(logn), stream);
  } else{
    d_twiddles = fill_twiddle_factors_array(n_twiddles, S::omega(logn), stream);
  }
  E * result = ntt_template < E, S > (arr, n, d_twiddles, n_twiddles, inverse, stream);
  for(int i = 0; i < n; i++){
    arr[i] = result[i]; 
  }
  cudaFreeAsync(d_twiddles, stream);
  cudaStreamSynchronize(stream);
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
template < typename T > __device__ __host__ void reverseOrder_batch(T * arr, uint32_t n, uint32_t logn, uint32_t task) {
  for (uint32_t i = 0; i < n; i++) {
    uint32_t reversed = reverseBits(i, logn);
    if (reversed > i) {
      T tmp = arr[task * n + i];
      arr[task * n + i] = arr[task * n + reversed];
      arr[task * n + reversed] = tmp;
    }
  }
}


/**
 * Cooley-Tukey butterfly kernel. 
 * @param arr array of objects of type E (elements). 
 * @param twiddles array of twiddle factors of type S (scalars). 
 * @param n size of arr. 
 * @param n_twiddles size of omegas.
 * @param m "pair distance" - indicate distance of butterflies inputs.
 * @param i Cooley-TUckey FFT stage number.
 * @param offset offset corr. to the specific taks (in batch).  
 */
template < typename E, typename S > __device__ __host__ void butterfly(E * arrReversed, S * omegas, uint32_t n, uint32_t n_omegas, uint32_t m, uint32_t i, uint32_t j, uint32_t offset) {
  uint32_t g = j * (n / m);
  uint32_t k = i + j + (m >> 1);
  E u = arrReversed[offset + i + j];
  E v = omegas[g * n_omegas / n] * arrReversed[offset + k];
  arrReversed[offset + i + j] = u + v;
  arrReversed[offset + k] = u - v;
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
__global__ void ntt_template_kernel(E *arr, uint32_t n, S *twiddles, uint32_t n_twiddles, uint32_t max_task, uint32_t s, bool rev)
{
  int task = blockIdx.x;
  int chunks = n / (blockDim.x * 2);

  if (task < max_task)
  {
    // flattened loop allows parallel processing
    uint32_t l = threadIdx.x;
    uint32_t loop_limit = blockDim.x;

    if (l < loop_limit)
    {
      uint32_t ntw_i = task % chunks;

      uint32_t shift_s = 1 << s;
      uint32_t shift2_s = 1 << (s + 1);
      uint32_t n_twiddles_div = n_twiddles >> (s + 1);

      l = ntw_i * blockDim.x + l; //to l from chunks to full

      uint32_t j = l & (shift_s - 1); // Equivalent to: l % (1 << s)
      uint32_t i = ((l / shift_s) * shift2_s) % n;
      uint32_t k = i + j + shift_s;

      uint32_t offset = (task / chunks) * n;
      E u = arr[offset + i + j];
      E v = rev ? arr[offset + k] : twiddles[j * n_twiddles_div] * arr[offset + k];
      arr[offset + i + j] = u + v;
      arr[offset + k] = u - v;
      if (rev)
        arr[offset + k] = twiddles[j * n_twiddles_div] * arr[offset + k];
    }
  }
}


/**
 * Cooley-Tukey NTT.
 * NOTE! this function assumes that d_twiddles are located in the device memory.
 * @param arr input array of type E (elements).
 * @param n length of arr.
 * @param logn log2(n).
 * @param max_task max count of parallel tasks.
 */
template <typename E, typename S>
__global__ void ntt_template_kernel_rev_ord(E *arr, uint32_t n, uint32_t logn, uint32_t max_task)
{
  int task = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (task < max_task)
  {
    reverseOrder_batch<E>(arr, n, logn, task);
  }
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
 template <typename E, typename S> uint32_t ntt_end2end_batch_template(E * arr, uint32_t arr_size, uint32_t n, bool inverse, cudaStream_t stream) {
  int batches = int(arr_size / n);
  uint32_t logn = uint32_t(log(n) / log(2));
  uint32_t n_twiddles = n; // n_twiddles is set to 4096 as BLS12_381::scalar_t::omega() is of that order. 
  size_t size_E = arr_size * sizeof(E);
  S * d_twiddles;
  if (inverse){
    d_twiddles = fill_twiddle_factors_array(n_twiddles, S::omega_inv(logn), stream);
  } else{
    d_twiddles = fill_twiddle_factors_array(n_twiddles, S::omega(logn), stream);
  }
  E * d_arr;
  cudaMallocAsync( & d_arr, size_E, stream);
  cudaMemcpyAsync(d_arr, arr, size_E, cudaMemcpyHostToDevice, stream);
  int NUM_THREADS = MAX_THREADS_BATCH;
  int NUM_BLOCKS = (batches + NUM_THREADS - 1) / NUM_THREADS;
  ntt_template_kernel_rev_ord<E, S><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(d_arr, n, logn, batches);

  NUM_THREADS = min(n / 2, MAX_THREADS_BATCH);
  int chunks = max(int((n / 2) / NUM_THREADS), 1);
  int total_tasks = batches * chunks;
  NUM_BLOCKS = total_tasks;

  //TODO: this loop also can be unrolled
  for (uint32_t s = 0; s < logn; s++)
  {
    ntt_template_kernel<E, S><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(d_arr, n, d_twiddles, n_twiddles, total_tasks, s, false);
    cudaStreamSynchronize(stream);
  }
  if (inverse == true)
  {
    NUM_THREADS = MAX_NUM_THREADS;
    NUM_BLOCKS = (arr_size + NUM_THREADS - 1) / NUM_THREADS;
    template_normalize_kernel < E, S > <<< NUM_THREADS, NUM_BLOCKS, 0, stream>>> (d_arr, arr_size, S::inv_log_size(logn));
  }
  cudaMemcpyAsync(arr, d_arr, size_E, cudaMemcpyDeviceToHost, stream);
  cudaFreeAsync(d_arr, stream);
  cudaFreeAsync(d_twiddles, stream);
  cudaStreamSynchronize(stream);
  return 0; 
}

#endif