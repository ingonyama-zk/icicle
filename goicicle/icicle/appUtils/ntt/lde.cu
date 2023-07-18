#ifndef LDE
#define LDE
#include <cuda.h>
#include "ntt.cuh"
#include "lde.cuh"
#include "../vector_manipulation/ve_mod_mult.cuh"

template < typename E, bool SUB > __global__ void add_sub_array(E* res, E* in1, E* in2, uint32_t n) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < n) {
      res[tid] = SUB ? in1[tid] - in2[tid] : in1[tid] + in2[tid];
    }
  }
  
  template <typename E>
  int sub_polys(E* d_out, E* d_in1, E* d_in2, unsigned n, cudaStream_t stream) {
    uint32_t NUM_THREADS = MAX_THREADS_BATCH;
    uint32_t NUM_BLOCKS = (n + NUM_THREADS - 1) / NUM_THREADS;
  
    add_sub_array <E, true> <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(d_out, d_in1, d_in2, n);
  
    return 0;
  }
  
  template <typename E>
  int add_polys(E* d_out, E* d_in1, E* d_in2, unsigned n, cudaStream_t stream) {
    uint32_t NUM_THREADS = MAX_THREADS_BATCH;
    uint32_t NUM_BLOCKS = (n + NUM_THREADS - 1) / NUM_THREADS;
  
    add_sub_array <E, false> <<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(d_out, d_in1, d_in2, n);
  
    return 0;
  }
  
/**
 * Interpolate a batch of polynomials from their evaluations on the same subgroup.
 * Note: this function does not preform any bit-reverse permutations on its inputs or outputs.
 * @param d_out The variable to write coefficients of the resulting polynomials into (the coefficients are in bit-reversed order if the evaluations weren't bit-reversed and vice-versa).
 * @param d_evaluations Input array of evaluations of all polynomials of type E (elements).
 * @param d_domain Domain on which the polynomials are evaluated. Must be a subgroup.
 * @param n Length of `d_domain` array, also equal to the number of evaluations of each polynomial.
 * @param batch_size The size of the batch; the length of `d_evaluations` is `n` * `batch_size`.
 */
template <typename E, typename S> int interpolate_batch(E * d_out, E * d_evaluations, S * d_domain, unsigned n, unsigned batch_size, bool coset, S * coset_powers, cudaStream_t stream) {
  cudaMemcpyAsync(d_out, d_evaluations, sizeof(E) * n * batch_size, cudaMemcpyDeviceToDevice, stream);
  ntt_inplace_batch_template(d_out, d_domain, n, batch_size, true, coset, coset_powers, stream, true);
  return 0;
}

/**
 * Interpolate a polynomial from its evaluations on a subgroup.
 * Note: this function does not preform any bit-reverse permutations on its inputs or outputs.
 * @param d_out The variable to write coefficients of the resulting polynomial into (the coefficients are in bit-reversed order if the evaluations weren't bit-reversed and vice-versa).
 * @param d_evaluations Input array of evaluations that have type E (elements).
 * @param d_domain Domain on which the polynomial is evaluated. Must be a subgroup.
 * @param n Length of `d_evaluations` and the size `d_domain` arrays (they should have equal length).
 */
template <typename E, typename S> int interpolate(E * d_out, E * d_evaluations, S * d_domain, unsigned n, bool coset, S * coset_powers, cudaStream_t stream) {
  return interpolate_batch <E, S> (d_out, d_evaluations, d_domain, n, 1, coset, coset_powers, stream);
}

template < typename E > __global__ void fill_array(E * arr, E val, uint32_t n) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < n) {
    arr[tid] = val;
  }
}

/**
 * Evaluate a batch of polynomials on the same coset.
 * @param d_out The evaluations of the polynomials on coset `u` * `d_domain`.
 * @param d_coefficients Input array of coefficients of all polynomials of type E (elements) to be evaluated in-place on a coset.
 * @param d_domain Domain on which the polynomials are evaluated (see `coset` flag). Must be a subgroup.
 * @param domain_size Length of `d_domain` array, on which the polynomial is computed.
 * @param n The number of coefficients, which might be different from `domain_size`.
 * @param batch_size The size of the batch; the length of `d_coefficients` is `n` * `batch_size`.
 * @param coset The flag that indicates whether to evaluate on a coset. If false, evaluate on a subgroup `d_domain`.
 * @param coset_powers If `coset` is true, a list of powers `[1, u, u^2, ..., u^{n-1}]` where `u` is the generator of the coset.
 */
template <typename E, typename S>
int evaluate_batch(E * d_out, E * d_coefficients, S * d_domain, unsigned domain_size, unsigned n, unsigned batch_size, bool coset, S * coset_powers, cudaStream_t stream) {
  uint32_t logn = uint32_t(log(domain_size) / log(2));
  if (domain_size > n) {
    // allocate and initialize an array of stream handles to parallelize data copying across batches
    cudaStream_t *memcpy_streams = (cudaStream_t *) malloc(batch_size * sizeof(cudaStream_t));
    for (unsigned i = 0; i < batch_size; i++)
    {
      cudaStreamCreate(&(memcpy_streams[i]));

      cudaMemcpyAsync(&d_out[i * domain_size], &d_coefficients[i * n], n * sizeof(E), cudaMemcpyDeviceToDevice, memcpy_streams[i]);
      uint32_t NUM_THREADS = MAX_THREADS_BATCH;
      uint32_t NUM_BLOCKS = (domain_size - n + NUM_THREADS - 1) / NUM_THREADS;
      fill_array <E> <<<NUM_BLOCKS, NUM_THREADS, 0, memcpy_streams[i]>>> (&d_out[i * domain_size + n], E::zero(), domain_size - n);

      cudaStreamSynchronize(memcpy_streams[i]);
      cudaStreamDestroy(memcpy_streams[i]);
    }
  } else
    cudaMemcpyAsync(d_out, d_coefficients, sizeof(E) * domain_size * batch_size, cudaMemcpyDeviceToDevice, stream);

  if (coset)
    batch_vector_mult(coset_powers, d_out, domain_size, batch_size, stream);
  
  S* _null = nullptr;
  ntt_inplace_batch_template(d_out, d_domain, domain_size, batch_size, false, false, _null, stream, true);
  return 0;
}

/**
 * Evaluate a polynomial on a coset.
 * Note: this function does not preform any bit-reverse permutations on its inputs or outputs, so the order of outputs is bit-reversed.
 * @param d_out The evaluations of the polynomial on coset `u` * `d_domain`.
 * @param d_coefficients Input array of coefficients of a polynomial of type E (elements).
 * @param d_domain Domain on which the polynomial is evaluated (see `coset` flag). Must be a subgroup.
 * @param domain_size Length of `d_domain` array, on which the polynomial is computed.
 * @param n The number of coefficients, which might be different from `domain_size`.
 * @param coset The flag that indicates whether to evaluate on a coset. If false, evaluate on a subgroup `d_domain`.
 * @param coset_powers If `coset` is true, a list of powers `[1, u, u^2, ..., u^{n-1}]` where `u` is the generator of the coset.
 */
template <typename E, typename S> 
int evaluate(E * d_out, E * d_coefficients, S * d_domain, unsigned domain_size, unsigned n, bool coset, S * coset_powers, cudaStream_t stream) {
  return evaluate_batch <E, S> (d_out, d_coefficients, d_domain, domain_size, n, 1, coset, coset_powers, stream);
}

template <typename S> 
int interpolate_scalars(S* d_out, S* d_evaluations, S* d_domain, unsigned n, cudaStream_t stream) {
  S* _null = nullptr;
  return interpolate(d_out, d_evaluations, d_domain, n, false, _null, stream);
}

template <typename S> 
int interpolate_scalars_batch(S* d_out, S* d_evaluations, S* d_domain, unsigned n, unsigned batch_size, cudaStream_t stream) {
  S* _null = nullptr;
  return interpolate_batch(d_out, d_evaluations, d_domain, n, batch_size, false, _null, stream);
}

template <typename E, typename S> 
int interpolate_points(E* d_out, E* d_evaluations, S* d_domain, unsigned n, cudaStream_t stream) {
  S* _null = nullptr;
  return interpolate(d_out, d_evaluations, d_domain, n, false, _null, stream);
}

template <typename E, typename S> 
int interpolate_points_batch(E* d_out, E* d_evaluations, S* d_domain, unsigned n, unsigned batch_size, cudaStream_t stream) {
  S* _null = nullptr;
  return interpolate_batch(d_out, d_evaluations, d_domain, n, batch_size, false, _null, stream);
}

template <typename S> 
int evaluate_scalars(S* d_out, S* d_coefficients, S* d_domain, unsigned domain_size, unsigned n, cudaStream_t stream) {
  S* _null = nullptr;
  return evaluate(d_out, d_coefficients, d_domain, domain_size, n, false, _null, stream);
}

template <typename S> 
int evaluate_scalars_batch(S* d_out, S* d_coefficients, S* d_domain, unsigned domain_size, unsigned n, unsigned batch_size, cudaStream_t stream) {
  S* _null = nullptr;
  return evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, false, _null, stream);
}

template <typename E, typename S> 
int evaluate_points(E* d_out, E* d_coefficients, S* d_domain, unsigned domain_size, unsigned n, cudaStream_t stream) {
  S* _null = nullptr;
  return evaluate(d_out, d_coefficients, d_domain, domain_size, n, false, _null, stream);
}

template <typename E, typename S> 
int evaluate_points_batch(E* d_out, E* d_coefficients, S* d_domain, 
                          unsigned domain_size, unsigned n, unsigned batch_size, cudaStream_t stream) {
  S* _null = nullptr;
  return evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, false, _null, stream);
}

template <typename S> 
int interpolate_scalars_on_coset(S* d_out, S* d_evaluations, S* d_domain,
                                 unsigned n, S* coset_powers, cudaStream_t stream) {
  return interpolate(d_out, d_evaluations, d_domain, n, true, coset_powers, stream);
}

template <typename S> 
int interpolate_scalars_on_coset_batch(S* d_out, S* d_evaluations, S* d_domain,
                                       unsigned n, unsigned batch_size, S* coset_powers, cudaStream_t stream) {
  return interpolate_batch(d_out, d_evaluations, d_domain, n, batch_size, true, coset_powers, stream);
}

template <typename S> 
int evaluate_scalars_on_coset(S* d_out, S* d_coefficients, S* d_domain, 
                              unsigned domain_size, unsigned n, S* coset_powers, cudaStream_t stream) {
  return evaluate(d_out, d_coefficients, d_domain, domain_size, n, true, coset_powers, stream);
}

template <typename E, typename S> 
int evaluate_scalars_on_coset_batch(S* d_out, S* d_coefficients, S* d_domain, unsigned domain_size, 
                                    unsigned n, unsigned batch_size, S* coset_powers, cudaStream_t stream) {
  return evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, true, coset_powers, stream);
}

template <typename E, typename S> 
int evaluate_points_on_coset(E* d_out, E* d_coefficients, S* d_domain, 
                             unsigned domain_size, unsigned n, S* coset_powers, cudaStream_t stream) {
  return evaluate(d_out, d_coefficients, d_domain, domain_size, n, true, coset_powers, stream);
}

template <typename E, typename S> 
int evaluate_points_on_coset_batch(E* d_out, E* d_coefficients, S* d_domain, unsigned domain_size,
                                   unsigned n, unsigned batch_size, S* coset_powers, cudaStream_t stream) {
  return evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, true, coset_powers, stream);
}
#endif