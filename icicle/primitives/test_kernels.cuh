#pragma once

#ifndef G2_DEFINED
#define G2_DEFINED

// TODO: change the curve depending on env variable
#include "../curves/bn254/curve_config.cuh"
#include "extension_field.cuh"
#include "projective.cuh"

#endif
#include "../curves/bls12_381/curve_config.cuh"
#include "projective.cuh"
#include "extension_field.cuh"

typedef Field<PARAMS_BLS12_381::fp_config> scalar_field;
typedef Field<PARAMS_BLS12_381::fq_config> base_field;
typedef Affine<base_field> affine;
static constexpr base_field b = base_field{ PARAMS_BLS12_381::weierstrass_b };
typedef Projective<base_field, scalar_field, b> proj;
typedef ExtensionField<PARAMS_BLS12_381::fq_config> base_extension_field;
typedef Affine<base_extension_field> g2_affine;
static constexpr base_extension_field b2 = base_extension_field{ base_field {PARAMS_BLS12_381::weierstrass_b_g2_re },  base_field {PARAMS_BLS12_381::weierstrass_b_g2_im }};
typedef Projective<base_extension_field, scalar_field, b2> g2_proj;

using namespace BN254;

template <class T1, class T2>
__global__ void add_elements_kernel(const T1* x, const T2* y, T1* result, const unsigned count)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  result[gid] = x[gid] + y[gid];
}

template <class T1, class T2>
int vec_add(const T1* x, const T2* y, T1* result, const unsigned count)
{
  add_elements_kernel<T1, T2><<<(count - 1) / 32 + 1, 32>>>(x, y, result, count);
  int error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

template <class T1, class T2>
__global__ void sub_elements_kernel(const T1* x, const T2* y, T1* result, const unsigned count)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  result[gid] = x[gid] - y[gid];
}

template <class T1, class T2>
int vec_sub(const T1* x, const T2* y, T1* result, const unsigned count)
{
  sub_elements_kernel<T1, T2><<<(count - 1) / 32 + 1, 32>>>(x, y, result, count);
  int error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

template <class T>
__global__ void neg_elements_kernel(const T* x, T* result, const unsigned count)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  result[gid] = T::neg(x[gid]);
}

template <class T>
int vec_neg(const T* x, T* result, const unsigned count)
{
  neg_elements_kernel<T><<<(count - 1) / 32 + 1, 32>>>(x, result, count);
  int error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

template <class F, class G>
__global__ void mul_elements_kernel(const F* x, const G* y, G* result, const unsigned count)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  result[gid] = x[gid] * y[gid];
}

template <class F, class G>
int vec_mul(const F* x, const G* y, G* result, const unsigned count)
{
  mul_elements_kernel<F, G><<<(count - 1) / 32 + 1, 32>>>(x, y, result, count);
  int error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

__global__ void inv_field_elements_kernel(const scalar_field_t* x, scalar_field_t* result, const unsigned count)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  result[gid] = scalar_field_t::inverse(x[gid]);
}

int field_vec_inv(const scalar_field_t* x, scalar_field_t* result, const unsigned count)
{
  inv_field_elements_kernel<<<(count - 1) / 32 + 1, 32>>>(x, result, count);
  int error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

__global__ void sqr_field_elements_kernel(const scalar_field_t* x, scalar_field_t* result, const unsigned count)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  result[gid] = scalar_field_t::sqr(x[gid]);
}

int field_vec_sqr(const scalar_field_t* x, scalar_field_t* result, const unsigned count)
{
  sqr_field_elements_kernel<<<(count - 1) / 32 + 1, 32>>>(x, result, count);
  int error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

template <class P, class A>
__global__ void to_affine_points_kernel(const P* x, A* result, const unsigned count)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  result[gid] = P::to_affine(x[gid]);
}

template <class P, class A>
int point_vec_to_affine(const P* x, A* result, const unsigned count)
{
  to_affine_points_kernel<P, A><<<(count - 1) / 32 + 1, 32>>>(x, result, count);
  int error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

__global__ void mp_mult_kernel(const scalar_field_t* x, const scalar_field_t* y, scalar_field_t::Wide* result)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  scalar_field_t::multiply_raw_device(x[gid].limbs_storage, y[gid].limbs_storage, result[gid].limbs_storage);
}

int mp_mult(const scalar_field_t* x, scalar_field_t* y, scalar_field_t::Wide* result)
{
  mp_mult_kernel<<<1, 32>>>(x, y, result);
  int error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

__global__ void mp_lsb_mult_kernel(const scalar_field_t* x, const scalar_field_t* y, scalar_field_t::Wide* result)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  scalar_field_t::multiply_lsb_raw_device(x[gid].limbs_storage, y[gid].limbs_storage, result[gid].limbs_storage);
}

int mp_lsb_mult(const scalar_field_t* x, scalar_field_t* y, scalar_field_t::Wide* result)
{
  mp_lsb_mult_kernel<<<1, 32>>>(x, y, result);
  int error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

__global__ void mp_msb_mult_kernel(const scalar_field_t* x, const scalar_field_t* y, scalar_field_t::Wide* result)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  scalar_field_t::multiply_msb_raw_device(x[gid].limbs_storage, y[gid].limbs_storage, result[gid].limbs_storage);
}

int mp_msb_mult(const scalar_field_t* x, scalar_field_t* y, scalar_field_t::Wide* result)
{
  mp_msb_mult_kernel<<<1, 1>>>(x, y, result);
  int error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

__global__ void ingo_mp_mult_kernel(const scalar_field_t* x, const scalar_field_t* y, scalar_field_t::Wide* result)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  scalar_field_t::ingo_multiply_raw_device(x[gid].limbs_storage, y[gid].limbs_storage, result[gid].limbs_storage);
}

int ingo_mp_mult(const scalar_field_t* x, scalar_field_t* y, scalar_field_t::Wide* result)
{
  ingo_mp_mult_kernel<<<1, 32>>>(x, y, result);
  int error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

__global__ void ingo_mp_msb_mult_kernel(const scalar_field_t* x, const scalar_field_t* y, scalar_field_t::Wide* result)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  scalar_field_t::ingo_msb_multiply_raw_device(x[gid].limbs_storage, y[gid].limbs_storage, result[gid].limbs_storage);
}

int ingo_mp_msb_mult(const scalar_field_t* x, scalar_field_t* y, scalar_field_t::Wide* result, const unsigned n)
{
  ingo_mp_msb_mult_kernel<<<1, n>>>(x, y, result);
  int error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

__global__ void ingo_mp_mod_mult_kernel(const scalar_field_t* x, const scalar_field_t* y, scalar_field_t* result)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  result[gid] = x[gid] * y[gid];
}

int ingo_mp_mod_mult(const scalar_field_t* x, scalar_field_t* y, scalar_field_t* result, const unsigned n)
{
  ingo_mp_mod_mult_kernel<<<1, n>>>(x, y, result);
  int error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}