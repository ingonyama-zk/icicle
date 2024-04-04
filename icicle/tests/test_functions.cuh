#pragma once

template <class T1, class T2, int N_REP>
__global__ void add_elements_kernel(const T1* x, const T2* y, T1* result, const unsigned count)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  T1 res = x[gid];
  T2 y_gid = y[gid];
  for (int i = 0; i < N_REP; i++)
    res = res + y_gid;
  result[gid] = res;
}

template <class T1, class T2, int N_REP = 1>
int vec_add(const T1* x, const T2* y, T1* result, const unsigned count)
{
  add_elements_kernel<T1, T2, N_REP><<<(count - 1) / 256 + 1, 256>>>(x, y, result, count);
  int error = cudaGetLastError();
  return (error || (N_REP > 1)) ? error : cudaDeviceSynchronize();
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
  sub_elements_kernel<T1, T2><<<(count - 1) / 256 + 1, 256>>>(x, y, result, count);
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
  neg_elements_kernel<T><<<(count - 1) / 256 + 1, 256>>>(x, result, count);
  int error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

template <class F, class G, int N_REP>
__global__ void mul_elements_kernel(const F* x, const G* y, G* result, const unsigned count)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  G res = x[gid];
  G y_gid = y[gid];
  for (int i = 0; i < N_REP; i++)
    res = res * y_gid;
  result[gid] = res;
}

template <class F, class G, int N_REP = 1>
int vec_mul(const F* x, const G* y, G* result, const unsigned count)
{
  mul_elements_kernel<F, G, N_REP><<<(count - 1) / 256 + 1, 256>>>(x, y, result, count);
  int error = cudaGetLastError();
  return (error || (N_REP > 1)) ? error : cudaDeviceSynchronize();
}

template <class F>
__global__ void inv_field_elements_kernel(const F* x, F* result, const unsigned count)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  result[gid] = F::inverse(x[gid]);
}

template <class F>
int field_vec_inv(const F* x, F* result, const unsigned count)
{
  inv_field_elements_kernel<<<(count - 1) / 256 + 1, 256>>>(x, result, count);
  int error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

template <class F, int N_REP>
__global__ void sqr_field_elements_kernel(const F* x, F* result, const unsigned count)
{
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  F x_gid = x[gid];
  for (int i = 0; i < N_REP; i++)
    x_gid = F::sqr(x_gid);
  result[gid] = x_gid;
}

template <class F, int N_REP = 1>
int field_vec_sqr(const F* x, F* result, const unsigned count)
{
  sqr_field_elements_kernel<F, N_REP><<<(count - 1) / 256 + 1, 256>>>(x, result, count);
  int error = cudaGetLastError();
  return (error || (N_REP > 1)) ? error : cudaDeviceSynchronize();
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
  to_affine_points_kernel<P, A><<<(count - 1) / 256 + 1, 256>>>(x, result, count);
  int error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

template <class T>
int device_populate_random(T* d_elements, unsigned n)
{
  T* h_elements = (T*)malloc(n * sizeof(T));
  for (unsigned i = 0; i < n; i++)
    h_elements[i] = T::rand_host();
  return cudaMemcpy(d_elements, h_elements, sizeof(T) * n, cudaMemcpyHostToDevice);
}

template <class T>
int device_set(T* d_elements, T el, unsigned n)
{
  T* h_elements = (T*)malloc(n * sizeof(T));
  for (unsigned i = 0; i < n; i++)
    h_elements[i] = el;
  return cudaMemcpy(d_elements, h_elements, sizeof(T) * n, cudaMemcpyHostToDevice);
}
