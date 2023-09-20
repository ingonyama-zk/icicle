#ifndef _BLS12_381_LDE
#define _BLS12_381_LDE
#include "../../appUtils/ntt/lde.cu"
#include "../../appUtils/ntt/ntt.cuh"
#include "../../appUtils/vector_manipulation/ve_mod_mult.cuh"
#include "../../utils/mont.cuh"
#include "curve_config.cuh"
#include <cuda.h>

extern "C" BLS12_381::scalar_t* build_domain_cuda_bls12_381(
  uint32_t domain_size, uint32_t logn, bool inverse, size_t device_id = 0, cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    if (inverse) {
      return fill_twiddle_factors_array(domain_size, BLS12_381::scalar_t::omega_inv(logn), stream);
    } else {
      return fill_twiddle_factors_array(domain_size, BLS12_381::scalar_t::omega(logn), stream);
    }
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return nullptr;
  }
}

extern "C" int
ntt_cuda_bls12_381(BLS12_381::scalar_t* arr, uint32_t n, bool inverse, size_t device_id = 0, cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return ntt_end2end_template<BLS12_381::scalar_t, BLS12_381::scalar_t>(
      arr, n, inverse, stream); // TODO: pass device_id
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());

    return -1;
  }
}

extern "C" int ecntt_cuda_bls12_381(
  BLS12_381::projective_t* arr, uint32_t n, bool inverse, size_t device_id = 0, cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return ntt_end2end_template<BLS12_381::projective_t, BLS12_381::scalar_t>(
      arr, n, inverse, stream); // TODO: pass device_id
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int ntt_batch_cuda_bls12_381(
  BLS12_381::scalar_t* arr,
  uint32_t arr_size,
  uint32_t batch_size,
  bool inverse,
  size_t device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return ntt_end2end_batch_template<BLS12_381::scalar_t, BLS12_381::scalar_t>(
      arr, arr_size, batch_size, inverse, stream); // TODO: pass device_id
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int ecntt_batch_cuda_bls12_381(
  BLS12_381::projective_t* arr,
  uint32_t arr_size,
  uint32_t batch_size,
  bool inverse,
  size_t device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return ntt_end2end_batch_template<BLS12_381::projective_t, BLS12_381::scalar_t>(
      arr, arr_size, batch_size, inverse, stream); // TODO: pass device_id
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int interpolate_scalars_cuda_bls12_381(
  BLS12_381::scalar_t* d_out,
  BLS12_381::scalar_t* d_evaluations,
  BLS12_381::scalar_t* d_domain,
  unsigned n,
  unsigned device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    BLS12_381::scalar_t* _null = nullptr;
    return interpolate(d_out, d_evaluations, d_domain, n, false, _null, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int interpolate_scalars_batch_cuda_bls12_381(
  BLS12_381::scalar_t* d_out,
  BLS12_381::scalar_t* d_evaluations,
  BLS12_381::scalar_t* d_domain,
  unsigned n,
  unsigned batch_size,
  size_t device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    BLS12_381::scalar_t* _null = nullptr;
    cudaStreamCreate(&stream);
    return interpolate_batch(d_out, d_evaluations, d_domain, n, batch_size, false, _null, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int interpolate_points_cuda_bls12_381(
  BLS12_381::projective_t* d_out,
  BLS12_381::projective_t* d_evaluations,
  BLS12_381::scalar_t* d_domain,
  unsigned n,
  size_t device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    BLS12_381::scalar_t* _null = nullptr;
    return interpolate(d_out, d_evaluations, d_domain, n, false, _null, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int interpolate_points_batch_cuda_bls12_381(
  BLS12_381::projective_t* d_out,
  BLS12_381::projective_t* d_evaluations,
  BLS12_381::scalar_t* d_domain,
  unsigned n,
  unsigned batch_size,
  size_t device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    BLS12_381::scalar_t* _null = nullptr;
    cudaStreamCreate(&stream);
    return interpolate_batch(d_out, d_evaluations, d_domain, n, batch_size, false, _null, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int evaluate_scalars_cuda_bls12_381(
  BLS12_381::scalar_t* d_out,
  BLS12_381::scalar_t* d_coefficients,
  BLS12_381::scalar_t* d_domain,
  unsigned domain_size,
  unsigned n,
  unsigned device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    BLS12_381::scalar_t* _null = nullptr;
    cudaStreamCreate(&stream);
    return evaluate(d_out, d_coefficients, d_domain, domain_size, n, false, _null, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int evaluate_scalars_batch_cuda_bls12_381(
  BLS12_381::scalar_t* d_out,
  BLS12_381::scalar_t* d_coefficients,
  BLS12_381::scalar_t* d_domain,
  unsigned domain_size,
  unsigned n,
  unsigned batch_size,
  size_t device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    BLS12_381::scalar_t* _null = nullptr;
    cudaStreamCreate(&stream);
    auto result_code = evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, false, _null, 0);
    cudaStreamDestroy(stream);
    return result_code;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int evaluate_points_cuda_bls12_381(
  BLS12_381::projective_t* d_out,
  BLS12_381::projective_t* d_coefficients,
  BLS12_381::scalar_t* d_domain,
  unsigned domain_size,
  unsigned n,
  size_t device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    BLS12_381::scalar_t* _null = nullptr;
    cudaStreamCreate(&stream);
    return evaluate(d_out, d_coefficients, d_domain, domain_size, n, false, _null, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int evaluate_points_batch_cuda_bls12_381(
  BLS12_381::projective_t* d_out,
  BLS12_381::projective_t* d_coefficients,
  BLS12_381::scalar_t* d_domain,
  unsigned domain_size,
  unsigned n,
  unsigned batch_size,
  size_t device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    BLS12_381::scalar_t* _null = nullptr;
    cudaStreamCreate(&stream);
    auto result_code =
      evaluate_batch(d_out, d_coefficients, d_domain, domain_size, n, batch_size, false, _null, stream);
    cudaStreamDestroy(stream);
    return result_code;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int evaluate_scalars_on_coset_cuda_bls12_381(
  BLS12_381::scalar_t* d_out,
  BLS12_381::scalar_t* d_coefficients,
  BLS12_381::scalar_t* d_domain,
  unsigned domain_size,
  unsigned n,
  BLS12_381::scalar_t* coset_powers,
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

extern "C" int evaluate_scalars_on_coset_batch_cuda_bls12_381(
  BLS12_381::scalar_t* d_out,
  BLS12_381::scalar_t* d_coefficients,
  BLS12_381::scalar_t* d_domain,
  unsigned domain_size,
  unsigned n,
  unsigned batch_size,
  BLS12_381::scalar_t* coset_powers,
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

extern "C" int evaluate_points_on_coset_cuda_bls12_381(
  BLS12_381::projective_t* d_out,
  BLS12_381::projective_t* d_coefficients,
  BLS12_381::scalar_t* d_domain,
  unsigned domain_size,
  unsigned n,
  BLS12_381::scalar_t* coset_powers,
  size_t device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(
      &stream);
    return evaluate(d_out, d_coefficients, d_domain, domain_size, n, true, coset_powers, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int evaluate_points_on_coset_batch_cuda_bls12_381(
  BLS12_381::projective_t* d_out,
  BLS12_381::projective_t* d_coefficients,
  BLS12_381::scalar_t* d_domain,
  unsigned domain_size,
  unsigned n,
  unsigned batch_size,
  BLS12_381::scalar_t* coset_powers,
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

extern "C" int ntt_inplace_batch_cuda_bls12_381(
  BLS12_381::scalar_t* d_inout,
  BLS12_381::scalar_t* d_twiddles,
  unsigned n,
  unsigned batch_size,
  bool inverse,
  size_t device_id = 0,
  cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    BLS12_381::scalar_t* _null = nullptr;
    ntt_inplace_batch_template(d_inout, d_twiddles, n, batch_size, inverse, false, _null, stream, true);
    return CUDA_SUCCESS;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int
reverse_order_scalars_cuda_bls12_381(BLS12_381::scalar_t* arr, int n, size_t device_id = 0, cudaStream_t stream = 0)
{
  try {
    uint32_t logn = uint32_t(log(n) / log(2));
    cudaStreamCreate(&stream);
    reverse_order(arr, n, logn, stream);
    return 0;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int reverse_order_scalars_batch_cuda_bls12_381(
  BLS12_381::scalar_t* arr, int n, int batch_size, size_t device_id = 0, cudaStream_t stream = 0)
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

extern "C" int
reverse_order_points_cuda_bls12_381(BLS12_381::projective_t* arr, int n, size_t device_id = 0, cudaStream_t stream = 0)
{
  try {
    uint32_t logn = uint32_t(log(n) / log(2));
    cudaStreamCreate(&stream);
    reverse_order(arr, n, logn, stream);
    return 0;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int sub_scalars_cuda_bls12_381(
  BLS12_381::scalar_t* d_out,
  BLS12_381::scalar_t* d_in1,
  BLS12_381::scalar_t* d_in2,
  unsigned n,
  cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return sub_polys(d_out, d_in1, d_in2, n, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int add_scalars_cuda_bls12_381(
  BLS12_381::scalar_t* d_out,
  BLS12_381::scalar_t* d_in1,
  BLS12_381::scalar_t* d_in2,
  unsigned n,
  cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return add_polys(d_out, d_in1, d_in2, n, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int to_montgomery_scalars_cuda_bls12_381(BLS12_381::scalar_t* d_inout, unsigned n, cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return to_montgomery(d_inout, n, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int from_montgomery_scalars_cuda_bls12_381(BLS12_381::scalar_t* d_inout, unsigned n, cudaStream_t stream = 0)
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
to_montgomery_proj_points_cuda_bls12_381(BLS12_381::projective_t* d_inout, unsigned n, cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return to_montgomery((BLS12_381::point_field_t*)d_inout, 3 * n, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int
from_montgomery_proj_points_cuda_bls12_381(BLS12_381::projective_t* d_inout, unsigned n, cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return from_montgomery((BLS12_381::point_field_t*)d_inout, 3 * n, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int
to_montgomery_aff_points_cuda_bls12_381(BLS12_381::affine_t* d_inout, unsigned n, cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return to_montgomery((BLS12_381::point_field_t*)d_inout, 2 * n, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int
from_montgomery_aff_points_cuda_bls12_381(BLS12_381::affine_t* d_inout, unsigned n, cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return from_montgomery((BLS12_381::point_field_t*)d_inout, 2 * n, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

#if defined(G2_DEFINED)
extern "C" int
to_montgomery_proj_points_g2_cuda_bls12_381(BLS12_381::g2_projective_t* d_inout, unsigned n, cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return to_montgomery((BLS12_381::point_field_t*)d_inout, 6 * n, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int
from_montgomery_proj_points_g2_cuda_bls12_381(BLS12_381::g2_projective_t* d_inout, unsigned n, cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return from_montgomery((BLS12_381::point_field_t*)d_inout, 6 * n, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int
to_montgomery_aff_points_g2_cuda_bls12_381(BLS12_381::g2_affine_t* d_inout, unsigned n, cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return to_montgomery((BLS12_381::point_field_t*)d_inout, 4 * n, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}

extern "C" int
from_montgomery_aff_points_g2_cuda_bls12_381(BLS12_381::g2_affine_t* d_inout, unsigned n, cudaStream_t stream = 0)
{
  try {
    cudaStreamCreate(&stream);
    return from_montgomery((BLS12_381::point_field_t*)d_inout, 4 * n, stream);
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what());
    return -1;
  }
}
#endif

extern "C" int reverse_order_points_batch_cuda_bls12_381(
  BLS12_381::projective_t* arr, int n, int batch_size, size_t device_id = 0, cudaStream_t stream = 0)
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