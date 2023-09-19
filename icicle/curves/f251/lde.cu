#ifndef _F251_LDE
#define _F251_LDE
#include "../../appUtils/ntt/lde.cu"
#include "../../appUtils/ntt/ntt.cuh"
#include "../../appUtils/vector_manipulation/ve_mod_mult.cuh"
#include "../../utils/mont.cuh"
#include "curve_config.cuh"
#include <cuda.h>

extern "C" F251::scalar_t* build_domain_cuda_f251(
  uint32_t domain_size, uint32_t logn, bool inverse, size_t device_id = 0, cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    if (inverse) {
      return fill_twiddle_factors_array(domain_size, F251::scalar_t::omega_inv(logn), stream);
    } else {
      return fill_twiddle_factors_array(domain_size, F251::scalar_t::omega(logn), stream);
    }
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return nullptr;
  }
}

extern "C" int ntt_cuda_f251(
  F251::scalar_t* arr, uint32_t n, bool inverse, size_t device_id = 0, cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return ntt_end2end_template<F251::scalar_t, F251::scalar_t>(arr, n, inverse, stream); // TODO: pass device_id
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());

    return -1;
  }
}

extern "C" int ntt_batch_cuda_f251(
  F251::scalar_t* arr,
  uint32_t arr_size,
  uint32_t batch_size,
  bool inverse,
  size_t device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return ntt_end2end_batch_template<F251::scalar_t, F251::scalar_t>(
      arr, arr_size, batch_size, inverse, stream); // TODO: pass device_id
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int interpolate_scalars_cuda_f251(
  F251::scalar_t* d_out,
  F251::scalar_t* d_evaluations,
  F251::scalar_t* d_domain,
  unsigned n,
  unsigned device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    F251::scalar_t* _null = nullptr;
    return interpolate(d_out, d_evaluations, d_domain, n, false, _null, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int interpolate_scalars_batch_cuda_f251(
  F251::scalar_t* d_out,
  F251::scalar_t* d_evaluations,
  F251::scalar_t* d_domain,
  unsigned n,
  unsigned batch_size,
  size_t device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    F251::scalar_t* _null = nullptr;
    cudaStreamCreate(&stream);
    return interpolate_batch(d_out, d_evaluations, d_domain, n, batch_size, false, _null, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int interpolate_scalars_on_coset_cuda_f251(
  F251::scalar_t* d_out,
  F251::scalar_t* d_evaluations,
  F251::scalar_t* d_domain,
  unsigned n,
  F251::scalar_t* coset_powers,
  unsigned device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    return interpolate(d_out, d_evaluations, d_domain, n, true, coset_powers, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int interpolate_scalars_batch_on_coset_cuda_f251(
  F251::scalar_t* d_out,
  F251::scalar_t* d_evaluations,
  F251::scalar_t* d_domain,
  unsigned n,
  unsigned batch_size,
  F251::scalar_t* coset_powers,
  size_t device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return interpolate_batch(d_out, d_evaluations, d_domain, n, batch_size, true, coset_powers, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int evaluate_scalars_cuda_f251(
  F251::scalar_t* d_out,
  F251::scalar_t* d_coefficients,
  F251::scalar_t* d_domain,
  unsigned domain_size,
  unsigned n,
  unsigned device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    F251::scalar_t* _null = nullptr;
    cudaStreamCreate(&stream);
    return evaluate(d_out, d_coefficients, d_domain, domain_size, n, false, _null, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int evaluate_scalars_batch_cuda_f251(
  F251::scalar_t* d_out,
  F251::scalar_t* d_coefficients,
  F251::scalar_t* d_domain,
  unsigned domain_size,
  unsigned n,
  unsigned batch_size,
  size_t device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    F251::scalar_t* _null = nullptr;
    cudaStreamCreate(&stream);
    return evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, false, _null, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int evaluate_scalars_on_coset_cuda_f251(
  F251::scalar_t* d_out,
  F251::scalar_t* d_coefficients,
  F251::scalar_t* d_domain,
  unsigned domain_size,
  unsigned n,
  F251::scalar_t* coset_powers,
  unsigned device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return evaluate(d_out, d_coefficients, d_domain, domain_size, n, true, coset_powers, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int evaluate_scalars_on_coset_batch_cuda_f251(
  F251::scalar_t* d_out,
  F251::scalar_t* d_coefficients,
  F251::scalar_t* d_domain,
  unsigned domain_size,
  unsigned n,
  unsigned batch_size,
  F251::scalar_t* coset_powers,
  size_t device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, true, coset_powers, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int ntt_inplace_batch_cuda_f251(
  F251::scalar_t* d_inout,
  F251::scalar_t* d_twiddles,
  unsigned n,
  unsigned batch_size,
  bool inverse,
  size_t device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    F251::scalar_t* _null = nullptr;
    ntt_inplace_batch_template(d_inout, d_twiddles, n, batch_size, inverse, false, _null, stream, true);
    return CUDA_SUCCESS; // TODO: we should implement this https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int ntt_inplace_coset_batch_cuda_f251(
  F251::scalar_t* d_inout,
  F251::scalar_t* d_twiddles,
  unsigned n,
  unsigned batch_size,
  bool inverse,
  bool is_coset,
  F251::scalar_t* coset,
  size_t device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    ntt_inplace_batch_template(d_inout, d_twiddles, n, batch_size, inverse, is_coset, coset, stream, true);
    return CUDA_SUCCESS; // TODO: we should implement this https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int sub_scalars_cuda_f251(
  F251::scalar_t* d_out, F251::scalar_t* d_in1, F251::scalar_t* d_in2, unsigned n, cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return sub_polys(d_out, d_in1, d_in2, n, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int add_scalars_cuda_f251(
  F251::scalar_t* d_out, F251::scalar_t* d_in1, F251::scalar_t* d_in2, unsigned n, cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return add_polys(d_out, d_in1, d_in2, n, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int to_montgomery_scalars_cuda_f251(F251::scalar_t* d_inout, unsigned n, cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return to_montgomery(d_inout, n, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int from_montgomery_scalars_cuda_f251(F251::scalar_t* d_inout, unsigned n, cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return from_montgomery(d_inout, n, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int
reverse_order_scalars_cuda_f251(F251::scalar_t* arr, int n, size_t device_id = 0, cudaStream_t stream = 0)
{
  try {
    uint32_t logn = uint32_t(log(n) / log(2));
    cudaStreamCreate(&stream);
    reverse_order(arr, n, logn, stream);
    cudaStreamSynchronize(stream);
    return 0;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int reverse_order_scalars_batch_cuda_f251(
  F251::scalar_t* arr, int n, int batch_size, size_t device_id = 0, cudaStream_t stream = 0)
{
  try {
    uint32_t logn = uint32_t(log(n) / log(2));
    cudaStreamCreate(&stream);
    reverse_order_batch(arr, n, logn, batch_size, stream);
    return 0;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}
#endif