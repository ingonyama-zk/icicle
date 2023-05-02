#include <cuda.h>
#include "ntt.cuh"
#include "../vector_manipulation/ve_mod_mult.cuh"
#include "lde.cuh"


/**
 * Interpolate a batch of polynomials from their evaluations on the same subgroup.
 * Note: this function does not preform any bit-reverse permutations on its inputs or outputs.
 * @param d_out The variable to write coefficients of the resulting polynomials into (the coefficients are in bit-reversed order if the evaluations weren't bit-reversed and vice-versa).
 * @param d_evaluations Input array of evaluations of all polynomials of type E (elements).
 * @param d_domain Domain on which the polynomials are evaluated. Must be a subgroup.
 * @param n Length of `d_domain` array, also equal to the number of evaluations of each polynomial.
 * @param batch_size The size of the batch; the length of `d_evaluations` is `n` * `batch_size`.
 */
template <typename E, typename S> int interpolate_batch(E * d_out, E * d_evaluations, S * d_domain, unsigned n, unsigned batch_size) {
  uint32_t logn = uint32_t(log(n) / log(2));
  cudaMemcpy(d_out, d_evaluations, sizeof(E) * n * batch_size, cudaMemcpyDeviceToDevice);
  
  int NUM_THREADS = min(n / 2, MAX_THREADS_BATCH);
  int NUM_BLOCKS = batch_size * max(int((n / 2) / NUM_THREADS), 1);
  for (uint32_t s = 0; s < logn; s++) //TODO: this loop also can be unrolled
  {
    ntt_template_kernel <E, S> <<<NUM_BLOCKS, NUM_THREADS>>>(d_out, n, d_domain, n, NUM_BLOCKS, s, false);
  }

  NUM_BLOCKS = (n * batch_size + NUM_THREADS - 1) / NUM_THREADS;
  template_normalize_kernel <E, S> <<<NUM_BLOCKS, NUM_THREADS>>> (d_out, n * batch_size, scalar_t::inv_log_size(logn));
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
template <typename E, typename S> int interpolate(E * d_out, E * d_evaluations, S * d_domain, unsigned n) {
  return interpolate_batch <E, S> (d_out, d_evaluations, d_domain, n, 1);
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
int evaluate_batch(E * d_out, E * d_coefficients, S * d_domain, unsigned domain_size, unsigned n, unsigned batch_size, bool coset, S * coset_powers) {
  uint32_t logn = uint32_t(log(domain_size) / log(2));
  if (domain_size > n) {
    // allocate and initialize an array of stream handles to parallelize data copying across batches
    cudaStream_t *memcpy_streams = (cudaStream_t *) malloc(batch_size * sizeof(cudaStream_t));
    for (int i = 0; i < batch_size; i++)
    {
      cudaStreamCreate(&(memcpy_streams[i]));

      cudaMemcpyAsync(&d_out[i * domain_size], &d_coefficients[i * n], n * sizeof(E), cudaMemcpyDeviceToDevice, memcpy_streams[i]);
      int NUM_THREADS = MAX_THREADS_BATCH;
      int NUM_BLOCKS = (domain_size - n + NUM_THREADS - 1) / NUM_THREADS;
      fill_array <E> <<<NUM_BLOCKS, NUM_THREADS, 0, memcpy_streams[i]>>> (&d_out[i * domain_size + n], E::zero(), domain_size - n);

      cudaStreamSynchronize(memcpy_streams[i]);
      cudaStreamDestroy(memcpy_streams[i]);
    }
  } else
    cudaMemcpy(d_out, d_coefficients, sizeof(E) * domain_size * batch_size, cudaMemcpyDeviceToDevice);

  if (coset)
    batch_vector_mult(coset_powers, d_out, domain_size, batch_size);

  int NUM_THREADS = min(domain_size / 2, MAX_THREADS_BATCH);
  int chunks = max(int((domain_size / 2) / NUM_THREADS), 1);
  int NUM_BLOCKS = batch_size * chunks;
  for (uint32_t s = 0; s < logn; s++) //TODO: this loop also can be unrolled
  {
    ntt_template_kernel <E, S> <<<NUM_BLOCKS, NUM_THREADS>>>(d_out, domain_size, d_domain, domain_size, batch_size * chunks, logn - s - 1, true);
  }
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
int evaluate(E * d_out, E * d_coefficients, S * d_domain, unsigned domain_size, unsigned n, bool coset, S * coset_powers) {
  return evaluate_batch <E, S> (d_out, d_coefficients, d_domain, domain_size, n, 1, coset, coset_powers);
}

int interpolate_scalars(scalar_t* d_out, scalar_t* d_evaluations, scalar_t* d_domain, unsigned n) {
  return interpolate(d_out, d_evaluations, d_domain, n);
}

int interpolate_scalars_batch(scalar_t* d_out, scalar_t* d_evaluations, scalar_t* d_domain, unsigned n, unsigned batch_size) {
  return interpolate_batch(d_out, d_evaluations, d_domain, n, batch_size);
}

int interpolate_points(projective_t* d_out, projective_t* d_evaluations, scalar_t* d_domain, unsigned n) {
  return interpolate(d_out, d_evaluations, d_domain, n);
}

int interpolate_points_batch(projective_t* d_out, projective_t* d_evaluations, scalar_t* d_domain, unsigned n, unsigned batch_size) {
  return interpolate_batch(d_out, d_evaluations, d_domain, n, batch_size);
}

int evaluate_scalars(scalar_t* d_out, scalar_t* d_coefficients, scalar_t* d_domain, unsigned domain_size, unsigned n) {
  scalar_t* _null = nullptr;
  return evaluate(d_out, d_coefficients, d_domain, domain_size, n, false, _null);
}

int evaluate_scalars_batch(scalar_t* d_out, scalar_t* d_coefficients, scalar_t* d_domain, unsigned domain_size, unsigned n, unsigned batch_size) {
  scalar_t* _null = nullptr;
  return evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, false, _null);
}

int evaluate_points(projective_t* d_out, projective_t* d_coefficients, scalar_t* d_domain, unsigned domain_size, unsigned n) {
  scalar_t* _null = nullptr;
  return evaluate(d_out, d_coefficients, d_domain, domain_size, n, false, _null);
}

int evaluate_points_batch(projective_t* d_out, projective_t* d_coefficients, scalar_t* d_domain, 
                          unsigned domain_size, unsigned n, unsigned batch_size) {
  scalar_t* _null = nullptr;
  return evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, false, _null);
}

int evaluate_scalars_on_coset(scalar_t* d_out, scalar_t* d_coefficients, scalar_t* d_domain, 
                              unsigned domain_size, unsigned n, scalar_t* coset_powers) {
  return evaluate(d_out, d_coefficients, d_domain, domain_size, n, true, coset_powers);
}

int evaluate_scalars_on_coset_batch(scalar_t* d_out, scalar_t* d_coefficients, scalar_t* d_domain, unsigned domain_size, 
                                    unsigned n, unsigned batch_size, scalar_t* coset_powers) {
  return evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, true, coset_powers);
}

int evaluate_points_on_coset(projective_t* d_out, projective_t* d_coefficients, scalar_t* d_domain, 
                             unsigned domain_size, unsigned n, scalar_t* coset_powers) {
  return evaluate(d_out, d_coefficients, d_domain, domain_size, n, true, coset_powers);
}

int evaluate_points_on_coset_batch(projective_t* d_out, projective_t* d_coefficients, scalar_t* d_domain, unsigned domain_size,
                                   unsigned n, unsigned batch_size, scalar_t* coset_powers) {
  return evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, true, coset_powers);
}

extern "C" scalar_t* build_domain_cuda(uint32_t domain_size, uint32_t logn, bool inverse, size_t device_id = 0)
{
    try
    {
        if (inverse) {
            return fill_twiddle_factors_array(domain_size, scalar_t::omega_inv(logn));
        } else {
            return fill_twiddle_factors_array(domain_size, scalar_t::omega(logn));
        }
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return nullptr;
    }
}

extern "C" int ntt_cuda(scalar_t *arr, uint32_t n, bool inverse, size_t device_id = 0)
{
    try
    {
        return ntt_end2end(arr, n, inverse); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        
        return -1;        
    }
}

extern "C" int ecntt_cuda(projective_t *arr, uint32_t n, bool inverse, size_t device_id = 0)
{
    try
    {
        return ecntt_end2end(arr, n, inverse); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int ntt_batch_cuda(scalar_t *arr, uint32_t arr_size, uint32_t batch_size, bool inverse, size_t device_id = 0)
{
    try
    {
        return ntt_end2end_batch(arr, arr_size, batch_size, inverse); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int ecntt_batch_cuda(projective_t *arr, uint32_t arr_size, uint32_t batch_size, bool inverse, size_t device_id = 0)
{
    try
    {
        return ecntt_end2end_batch(arr, arr_size, batch_size, inverse); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int interpolate_scalars_cuda(scalar_t* d_out, scalar_t *d_evaluations, scalar_t *d_domain, unsigned n, unsigned device_id = 0)
{
    try
    {
        return interpolate_scalars(d_out, d_evaluations, d_domain, n); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int interpolate_scalars_batch_cuda(scalar_t* d_out, scalar_t* d_evaluations, scalar_t* d_domain, unsigned n,
                                              unsigned batch_size, size_t device_id = 0)
{
    try
    {
        return interpolate_scalars_batch(d_out, d_evaluations, d_domain, n, batch_size); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int interpolate_points_cuda(projective_t* d_out, projective_t *d_evaluations, scalar_t *d_domain, unsigned n, size_t device_id = 0)
{
    try
    {
        return interpolate_points(d_out, d_evaluations, d_domain, n); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int interpolate_points_batch_cuda(projective_t* d_out, projective_t* d_evaluations, scalar_t* d_domain,
                                             unsigned n, unsigned batch_size, size_t device_id = 0)
{
    try
    {
        return interpolate_points_batch(d_out, d_evaluations, d_domain, n, batch_size); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_scalars_cuda(scalar_t* d_out, scalar_t *d_coefficients, scalar_t *d_domain, 
                                     unsigned domain_size, unsigned n, unsigned device_id = 0)
{
    try
    {
        return evaluate_scalars(d_out, d_coefficients, d_domain, domain_size, n); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_scalars_batch_cuda(scalar_t* d_out, scalar_t* d_coefficients, scalar_t* d_domain, unsigned domain_size,
                                           unsigned n, unsigned batch_size, size_t device_id = 0)
{
    try
    {
        return evaluate_scalars_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_points_cuda(projective_t* d_out, projective_t *d_coefficients, scalar_t *d_domain, 
                                    unsigned domain_size, unsigned n, size_t device_id = 0)
{
    try
    {
        return evaluate_points(d_out, d_coefficients, d_domain, domain_size, n); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_points_batch_cuda(projective_t* d_out, projective_t* d_coefficients, scalar_t* d_domain, unsigned domain_size,
                                          unsigned n, unsigned batch_size, size_t device_id = 0)
{
    try
    {
        return evaluate_points_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_scalars_on_coset_cuda(scalar_t* d_out, scalar_t *d_coefficients, scalar_t *d_domain, unsigned domain_size,
                                              unsigned n, scalar_t *coset_powers, unsigned device_id = 0)
{
    try
    {
        return evaluate_scalars_on_coset(d_out, d_coefficients, d_domain, domain_size, n, coset_powers); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_scalars_on_coset_batch_cuda(scalar_t* d_out, scalar_t* d_coefficients, scalar_t* d_domain, unsigned domain_size, 
                                                    unsigned n, unsigned batch_size, scalar_t *coset_powers, size_t device_id = 0)
{
    try
    {
        return evaluate_scalars_on_coset_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, coset_powers); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_points_on_coset_cuda(projective_t* d_out, projective_t *d_coefficients, scalar_t *d_domain, unsigned domain_size,
                                             unsigned n, scalar_t *coset_powers, size_t device_id = 0)
{
    try
    {
        return evaluate_points_on_coset(d_out, d_coefficients, d_domain, domain_size, n, coset_powers); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int evaluate_points_on_coset_batch_cuda(projective_t* d_out, projective_t* d_coefficients, scalar_t* d_domain, unsigned domain_size, 
                                                   unsigned n, unsigned batch_size, scalar_t *coset_powers, size_t device_id = 0)
{
    try
    {
        return evaluate_points_on_coset_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, coset_powers); // TODO: pass device_id
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int reverse_order_scalars_cuda(scalar_t* arr, int n, size_t device_id = 0)
{
    try
    {
        uint32_t logn = uint32_t(log(n) / log(2));
        reverse_order(arr, n, logn);
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int reverse_order_scalars_batch_cuda(scalar_t* arr, int n, int batch_size, size_t device_id = 0)
{
    try
    {
        uint32_t logn = uint32_t(log(n) / log(2));
        reverse_order_batch(arr, n, logn, batch_size);
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int reverse_order_points_cuda(projective_t* arr, int n, size_t device_id = 0)
{
    try
    {
        uint32_t logn = uint32_t(log(n) / log(2));
        reverse_order(arr, n, logn);
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}

extern "C" int reverse_order_points_batch_cuda(projective_t* arr, int n, int batch_size, size_t device_id = 0)
{
    try
    {
        uint32_t logn = uint32_t(log(n) / log(2));
        reverse_order_batch(arr, n, logn, batch_size);
        return 0;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}
