#pragma once

#ifndef G2_DEFINED
#define G2_DEFINED

#include "../curves/curve_config.cuh"
#include "extension_field.cuh"
#include "projective.cuh"

#endif

using namespace curve_config;

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

__global__ void inv_field_elements_kernel(const scalar_t* x, scalar_t* result, const unsigned count)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  result[gid] = scalar_t::inverse(x[gid]);
}

int field_vec_inv(const scalar_t* x, scalar_t* result, const unsigned count)
{
  inv_field_elements_kernel<<<(count - 1) / 32 + 1, 32>>>(x, result, count);
  int error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

__global__ void sqr_field_elements_kernel(const scalar_t* x, scalar_t* result, const unsigned count)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  result[gid] = scalar_t::sqr(x[gid]);
}

int field_vec_sqr(const scalar_t* x, scalar_t* result, const unsigned count)
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
