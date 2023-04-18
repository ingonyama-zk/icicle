#pragma once

#include "ntt.cuh"
#include "../vector_manipulation/ve_mod_mult.cuh"


/**
 * Interpolate a polynomial from its evaluations on a subgroup.
 * Note: this function does not preform any bit-reverse permutations on its inputs or outputs.
 * @param d_evaluations Input array of evaluations that have type E (elements).
 * @param d_domain Domain on which the polynomial is evaluated. Must be a subgroup.
 * @param n Length of `d_evaluations` and the size `d_domain` arrays (they should have equal length).
 * @returns The coefficients of the resulting polynomial in bit-reversed order if the evaluations weren't bit-reversed and vice-versa.
 */
template < typename E, typename S > E * interpolate(E * d_evaluations, S * d_domain, unsigned n) {
  // TODO: call batch interpolate with batch size 1 once batch interpolation is as efficient as this function
  E * d_coefficients;
  uint32_t logn = uint32_t(log(n) / log(2));
  cudaMalloc(&d_coefficients, sizeof(E) * n);
  cudaMemcpy(d_coefficients, d_evaluations, sizeof(E) * n, cudaMemcpyDeviceToDevice);
  template_ntt_on_device_memory < E, S > (d_coefficients, n, logn, d_domain, n);
  int NUM_THREADS = MAX_THREADS_BATCH;
  int NUM_BLOCKS = (n + NUM_THREADS - 1) / NUM_THREADS;
  template_normalize_kernel < E, S > <<< NUM_BLOCKS, NUM_THREADS >>> (d_coefficients, n, S::inv_log_size(logn));
  return d_coefficients;
}

/**
 * Interpolate a batch of polynomials from their evaluations on the same subgroup.
 * Note: this function does not preform any bit-reverse permutations on its inputs or outputs.
 * @param d_evaluations Input array of evaluations of all polynomials of type E (elements).
 * @param d_domain Domain on which the polynomials are evaluated. Must be a subgroup.
 * @param n Length of `d_domain` array, also equal to the number of evaluations of each polynomial.
 * @param batch_size The size of the batch; the length of `d_evaluations` is `n` * `batch_size`.
 * @returns The coefficients of the resulting polynomials in bit-reversed order if the evaluations weren't bit-reversed and vice-versa.
 */
template < typename E, typename S > E * interpolate_batch(E * d_evaluations, S * d_domain, unsigned n, unsigned batch_size) {
  E * d_coefficients;
  uint32_t logn = uint32_t(log(n) / log(2));
  cudaMalloc(&d_coefficients, sizeof(E) * n * batch_size);
  cudaMemcpy(d_coefficients, d_evaluations, sizeof(E) * n * batch_size, cudaMemcpyDeviceToDevice);
  int NUM_THREADS = MAX_THREADS_BATCH;
  int NUM_BLOCKS = (batch_size + NUM_THREADS - 1) / NUM_THREADS;
  ntt_template_kernel < E, S > <<< NUM_BLOCKS, NUM_THREADS >>>(d_coefficients, n, logn, d_domain, n, batch_size);
  NUM_BLOCKS = (n * batch_size + NUM_THREADS - 1) / NUM_THREADS;
  template_normalize_kernel < E, S > <<< NUM_BLOCKS, NUM_THREADS >>> (d_coefficients, n * batch_size, scalar_t::inv_log_size(logn));
  return d_coefficients;
}

template < typename E > __global__ void fill_array(E * arr, E val, uint32_t n) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < n) {
    arr[tid] = val;
  }
}

/**
 * Evaluate a polynomial on a coset.
 * Note: this function does not preform any bit-reverse permutations on its inputs or outputs, so the order of outputs is bit-reversed.
 * @param d_coefficients Input array of coefficients of a polynomial of type E (elements).
 * @param d_domain Domain on which the polynomial is evaluated (see `coset` flag). Must be a subgroup.
 * @param domain_size Length of `d_domain` array, on which the polynomial is computed.
 * @param n The number of coefficients, which might be different from `domain_size`.
 * @param coset The flag that indicates whether to evaluate on a coset. If false, evaluate on a subgroup `d_domain`.
 * @param coset_powers If `coset` is true, a list of powers `[1, u, u^2, ..., u^{n-1}]` where `u` is the generator of the coset.
 * @returns The evaluations of the polynomials on coset `u` * `d_domain`.
 */
template < typename E, typename S > 
E * evaluate(E * d_coefficients, S * d_domain, unsigned domain_size, unsigned n, bool coset, S * coset_powers) {
  uint32_t logn = uint32_t(log(domain_size) / log(2));
  E * d_evaluations;
  cudaMalloc(&d_evaluations, sizeof(E) * domain_size);
  if (domain_size > n) {
    cudaMemcpy(d_evaluations, d_coefficients, n * sizeof(E), cudaMemcpyDeviceToDevice);
    int NUM_THREADS = MAX_THREADS_BATCH;
    int NUM_BLOCKS = (domain_size - n + NUM_THREADS - 1) / NUM_THREADS;
    fill_array <E> <<< NUM_BLOCKS, NUM_THREADS >>> (&d_evaluations[n], E::zero(), domain_size - n);
  } else
    cudaMemcpy(d_evaluations, d_coefficients, sizeof(E) * domain_size, cudaMemcpyDeviceToDevice);

  if (coset)
    batch_vector_mult(coset_powers, d_evaluations, domain_size, 1);

  reverse_order(d_evaluations, domain_size, logn);

  template_ntt_on_device_memory < E, S > (d_evaluations, domain_size, logn, d_domain, domain_size);
  return d_evaluations;
}

/**
 * Evaluate a batch of polynomials on the same coset.
 * Note: this function does not preform any bit-reverse permutations on its inputs or outputs, so the order of outputs is bit-reversed.
 * @param d_coefficients Input array of coefficients of all polynomials of type E (elements).
 * @param d_domain Domain on which the polynomials are evaluated (see `coset` flag). Must be a subgroup.
 * @param domain_size Length of `d_domain` array, on which the polynomial is computed.
 * @param n The number of coefficients, which might be different from `domain_size`.
 * @param batch_size The size of the batch; the length of `d_coefficients` is `n` * `batch_size`.
 * @param coset The flag that indicates whether to evaluate on a coset. If false, evaluate on a subgroup `d_domain`.
 * @param coset_powers If `coset` is true, a list of powers `[1, u, u^2, ..., u^{n-1}]` where `u` is the generator of the coset.
 * @returns The evaluations of the polynomials on coset `u` * `d_domain`.
 */
template < typename E, typename S > 
E * evaluate_batch(E * d_coefficients, S * d_domain, unsigned domain_size, unsigned n, unsigned batch_size, bool coset, S * coset_powers) {
  uint32_t logn = uint32_t(log(domain_size) / log(2));
  E * d_evaluations;
  cudaMalloc(&d_evaluations, sizeof(E) * domain_size * batch_size);
  if (domain_size > n) {
    // allocate and initialize an array of stream handles to parallelize data copying across batches
    cudaStream_t *memcpy_streams = (cudaStream_t *) malloc(batch_size * sizeof(cudaStream_t));
    for (int i = 0; i < batch_size; i++)
    {
      cudaStreamCreate(&(memcpy_streams[i]));

      cudaMemcpyAsync(&d_evaluations[i * domain_size], &d_coefficients[i * n], n * sizeof(E), cudaMemcpyDeviceToDevice, memcpy_streams[i]);
      int NUM_THREADS = MAX_THREADS_BATCH;
      int NUM_BLOCKS = (domain_size - n + NUM_THREADS - 1) / NUM_THREADS;
      fill_array <E> <<< NUM_BLOCKS, NUM_THREADS, 0, memcpy_streams[i] >>> (&d_evaluations[i * domain_size + n], E::zero(), domain_size - n);

      cudaStreamSynchronize(memcpy_streams[i]);
      cudaStreamDestroy(memcpy_streams[i]);
    }
  } else
    cudaMemcpy(d_evaluations, d_coefficients, sizeof(E) * domain_size * batch_size, cudaMemcpyDeviceToDevice);

  // reverse_order_batch(d_evaluations, domain_size, logn, batch_size);

  if (coset)
    batch_vector_mult(coset_powers, d_evaluations, domain_size, batch_size);

  int NUM_THREADS = MAX_THREADS_BATCH;
  int NUM_BLOCKS = (batch_size + NUM_THREADS - 1) / NUM_THREADS;
  ntt_template_kernel < E, S > <<< NUM_BLOCKS, NUM_THREADS >>>(d_evaluations, domain_size, logn, d_domain, domain_size, batch_size);
  return d_evaluations;
}

scalar_t* interpolate_scalars(scalar_t* d_evaluations, scalar_t* d_domain, unsigned n) {
  return interpolate(d_evaluations, d_domain, n);
}

scalar_t* interpolate_scalars_batch(scalar_t* d_evaluations, scalar_t* d_domain, unsigned n, unsigned batch_size) {
  return interpolate_batch(d_evaluations, d_domain, n, batch_size);
}

projective_t* interpolate_points(projective_t* d_evaluations, scalar_t* d_domain, unsigned n) {
  return interpolate(d_evaluations, d_domain, n);
}

projective_t* interpolate_points_batch(projective_t* d_evaluations, scalar_t* d_domain, unsigned n, unsigned batch_size) {
  return interpolate_batch(d_evaluations, d_domain, n, batch_size);
}

scalar_t* evaluate_scalars(scalar_t* d_coefficients, scalar_t* d_domain, unsigned domain_size, unsigned n) {
  scalar_t* _null = nullptr;
  return evaluate(d_coefficients, d_domain, domain_size, n, false, _null);
}

scalar_t* evaluate_scalars_batch(scalar_t* d_coefficients, scalar_t* d_domain, unsigned domain_size, unsigned n, unsigned batch_size) {
  scalar_t* _null = nullptr;
  return evaluate_batch(d_coefficients, d_domain, domain_size, n, batch_size, false, _null);
}

projective_t* evaluate_points(projective_t* d_coefficients, scalar_t* d_domain, unsigned domain_size, unsigned n) {
  scalar_t* _null = nullptr;
  return evaluate(d_coefficients, d_domain, domain_size, n, false, _null);
}

projective_t* evaluate_points_batch(projective_t* d_coefficients, scalar_t* d_domain, 
                                    unsigned domain_size, unsigned n, unsigned batch_size) {
  scalar_t* _null = nullptr;
  return evaluate_batch(d_coefficients, d_domain, domain_size, n, batch_size, false, _null);
}

scalar_t* evaluate_scalars_on_coset(scalar_t* d_coefficients, scalar_t* d_domain, 
                                    unsigned domain_size, unsigned n, scalar_t* coset_powers) {
  return evaluate(d_coefficients, d_domain, domain_size, n, true, coset_powers);
}

scalar_t* evaluate_scalars_on_coset_batch(scalar_t* d_coefficients, scalar_t* d_domain, unsigned domain_size, 
                                          unsigned n, unsigned batch_size, scalar_t* coset_powers) {
  return evaluate_batch(d_coefficients, d_domain, domain_size, n, batch_size, true, coset_powers);
}

projective_t* evaluate_points_on_coset(projective_t* d_coefficients, scalar_t* d_domain, 
                                       unsigned domain_size, unsigned n, scalar_t* coset_powers) {
  return evaluate(d_coefficients, d_domain, domain_size, n, true, coset_powers);
}

projective_t* evaluate_points_on_coset_batch(projective_t* d_coefficients, scalar_t* d_domain, unsigned domain_size,
                                             unsigned n, unsigned batch_size, scalar_t* coset_powers) {
  return evaluate_batch(d_coefficients, d_domain, domain_size, n, batch_size, true, coset_powers);
}
