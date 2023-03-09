#include <bits/stdc++.h>

#include "../../curves/curve_config.cuh"

const uint32_t MAX_NUM_THREADS = 1024;


/**
 * Copy twiddle factors array to device (returns a pointer to the device allocated array).
 * @param twiddles input empty array. 
 * @param n_twiddles length of twiddle factors. 
 */
 scalar_t * copy_twiddle_factors_to_device(scalar_t * twiddles, uint32_t n_twiddles) {
  size_t size_twiddles = n_twiddles * sizeof(scalar_t);
  scalar_t * d_twiddles;
  cudaMalloc( & d_twiddles, size_twiddles);
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
__global__ void twiddle_factors_kernel(scalar_t * d_twiddles, uint32_t n_twiddles, scalar_t omega) {
  for (uint32_t i = 0; i < n_twiddles; i++) {
    d_twiddles[i] = scalar_t::zero();
  }
  d_twiddles[0] = scalar_t::one();
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
void fill_twiddle_factors_array(scalar_t * twiddles, uint32_t n_twiddles, scalar_t omega) {
  size_t size_twiddles = n_twiddles * sizeof(scalar_t);
  scalar_t * d_twiddles;
  cudaMalloc( & d_twiddles, size_twiddles);
  cudaMemcpy(d_twiddles, twiddles, size_twiddles, cudaMemcpyHostToDevice);
  twiddle_factors_kernel <<< 1, 1 >>> (d_twiddles, n_twiddles, omega);
  cudaMemcpy(twiddles, d_twiddles, size_twiddles, cudaMemcpyDeviceToHost);
  cudaFree(d_twiddles);
}

/**
 * Returens the bit reversed order of a number. 
 * for example: on inputs num = 6 (110 in binary) and logn = 3
 * the function should return 3 (011 in binary.)
 * @param num some number with bit representation of size logn.
 * @param logn length of bit representation of num.
 * @return bit reveresed order or num.
 */
__device__ __host__ uint32_t reverseBits(uint32_t num, uint32_t logn) {
  unsigned int reverse_num = 0;
  int i;
  for (i = 0; i < logn; i++) {
    if ((num & (1 << i))) reverse_num |= 1 << ((logn - 1) - i);
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
template < typename T > T * template_reverse_order(T * arr, uint32_t n, uint32_t logn) {
  T * arrReversed = new T[n];
  for (uint32_t i = 0; i < n; i++) {
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
template < typename E, typename S > __global__ void template_butterfly_kernel(E * arr, S * twiddles, uint32_t n, uint32_t n_twiddles, uint32_t m, uint32_t i, uint32_t max_thread_num) {
  int j = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (j < max_thread_num) {
    uint32_t g = j * (n / m);
    uint32_t k = i + j + (m >> 1);
    E u = S::one() * arr[i + j];
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
template < typename E, typename S > __global__ void template_normalize_kernel(E * arr, E * res, uint32_t n, S scalar) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < n) {
    res[tid] = scalar * arr[tid];
  }
}

/**
 * Cooley-Tuckey NTT. 
 * NOTE! this function assumes that d_arr and d_twiddles are located in the device memory.
 * @param d_arr input array of type E (elements) allocated on the device memory. 
 * @param n length of d_arr (must be either 4, 256, 512, 4096).
 * @param logn log(n). 
 * @param d_twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2). 
 * @param n_twiddles length of d_twiddles. 
 */
template < typename E, typename S > void template_ntt_on_device_memory(E * d_arr, uint32_t n, uint32_t logn, S * d_twiddles, uint32_t n_twiddles) {
  uint32_t m = 2;
  for (uint32_t s = 0; s < logn; s++) {
    for (uint32_t i = 0; i < n; i += m) {
        int shifted_m = m >> 1;
        int number_of_threads = MAX_NUM_THREADS ^ ((shifted_m ^ MAX_NUM_THREADS) & -(shifted_m < MAX_NUM_THREADS));
        int number_of_blocks = shifted_m / MAX_NUM_THREADS + 1;
        template_butterfly_kernel < E, S > <<< number_of_threads, number_of_blocks >>> (d_arr, d_twiddles, n, n_twiddles, m, i, m >> 1);
    }
    m <<= 1;
  }
}

/**
 * Cooley-Tuckey NTT. 
 * NOTE! this function assumes that d_twiddles are located in the device memory.
 * @param arr input array of type E (elements). 
 * @param n length of d_arr (must be either 4, 256, 512, 4096).
 * @param d_twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2). 
 * @param n_twiddles length of d_twiddles. 
 * @param inverse indicate if the result array should be normalized by n^(-1). 
 */
template < typename E, typename S > E * ntt_template(E * arr, uint32_t n, S * d_twiddles, uint32_t n_twiddles, bool inverse) {
  uint32_t logn = uint32_t(log(n) / log(2));
  size_t size_E = n * sizeof(E);
  E * arrReversed = template_reverse_order < E > (arr, n, logn);
  E * d_arrReversed;
  cudaMalloc( & d_arrReversed, size_E);
  cudaMemcpy(d_arrReversed, arrReversed, size_E, cudaMemcpyHostToDevice);
  template_ntt_on_device_memory < E, S > (d_arrReversed, n, logn, d_twiddles, n_twiddles);
  if (inverse == true) {
    int NUM_THREADS = MAX_NUM_THREADS;
    int NUM_BLOCKS = (n + NUM_THREADS - 1) / NUM_THREADS;
    template_normalize_kernel < E, S > <<< NUM_THREADS, NUM_BLOCKS >>> (d_arrReversed, d_arrReversed, n, S::inv_size(n));
  }
  cudaMemcpy(arrReversed, d_arrReversed, size_E, cudaMemcpyDeviceToHost);
  cudaFree(d_arrReversed);
  return arrReversed;
}

/**
 * Cooley-Tuckey Elliptic Curve NTT. 
 * NOTE! this function assumes that d_twiddles are located in the device memory.
 * @param arr input array of type projective_t. 
 * @param n length of d_arr (must be either 4, 256, 512, 4096).
 * @param d_twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2). 
 * @param n_twiddles length of d_twiddles. 
 * @param inverse indicate if the result array should be normalized by n^(-1). 
 */
projective_t * ecntt(projective_t * arr, uint32_t n, scalar_t * d_twiddles, uint32_t n_twiddles, bool inverse) {
  return ntt_template < projective_t, scalar_t > (arr, n, d_twiddles, n_twiddles, inverse);
}

/**
 * Cooley-Tuckey (scalar) NTT. 
 * NOTE! this function assumes that d_twiddles are located in the device memory.
 * @param arr input array of type scalar_t. 
 * @param n length of d_arr (must be either 4, 256, 512, 4096).
 * @param d_twiddles twiddle factors of type S (scalars) array allocated on the device memory (must be a power of 2). 
 * @param n_twiddles length of d_twiddles. 
 * @param inverse indicate if the result array should be normalized by n^(-1). 
 */
scalar_t * ntt(scalar_t * arr, uint32_t n, scalar_t * d_twiddles, uint32_t n_twiddles, bool inverse) {
  return ntt_template < scalar_t, scalar_t > (arr, n, d_twiddles, n_twiddles, inverse);
}


/**
 * Cooley-Tuckey (scalar) NTT. 
 * @param arr input array of type scalar_t. 
 * @param n length of d_arr (must be either 4, 256, 512, 4096).
 * @param inverse indicate if the result array should be normalized by n^(-1). 
 */
 int ntt_end2end(scalar_t * arr, uint32_t n, bool inverse) {
  uint32_t n_twiddles = 4096; // n_twiddles is set to 4096 as scalar_t::omega() is of that order. 
  scalar_t * twiddles = new scalar_t[n_twiddles];
  if (inverse){
    fill_twiddle_factors_array(twiddles, n_twiddles, scalar_t::omega_inv());
  } else{
    fill_twiddle_factors_array(twiddles, n_twiddles, scalar_t::omega());
  }
  scalar_t * d_twiddles = copy_twiddle_factors_to_device(twiddles, n_twiddles);
  scalar_t * result = ntt_template < scalar_t, scalar_t > (arr, n, d_twiddles, n_twiddles, inverse);
  for(int i = 0; i < n; i++){
    arr[i] = result[i]; 
  }
  cudaFree(d_twiddles);
  return 0; 
}


/**
 * Cooley-Tuckey (scalar) NTT. 
 * @param arr input array of type scalar_t. 
 * @param n length of d_arr (must be either 4, 256, 512, 4096).
 * @param inverse indicate if the result array should be normalized by n^(-1). 
 */
 int ecntt_end2end(projective_t * arr, uint32_t n, bool inverse) {
  uint32_t n_twiddles = 4096; // n_twiddles is set to 4096 as scalar_t::omega() is of that order. 
  scalar_t * twiddles = new scalar_t[n_twiddles];
  if (inverse){
    fill_twiddle_factors_array(twiddles, n_twiddles, scalar_t::omega_inv());
  } else{
    fill_twiddle_factors_array(twiddles, n_twiddles, scalar_t::omega());
  }
  scalar_t * d_twiddles = copy_twiddle_factors_to_device(twiddles, n_twiddles);
  projective_t * result = ntt_template < projective_t, scalar_t > (arr, n, d_twiddles, n_twiddles, inverse);
  for(int i = 0; i < n; i++){
    arr[i] = result[i]; 
  }
  cudaFree(d_twiddles);
  return 0; 
}

