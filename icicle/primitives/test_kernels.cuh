#pragma once

template <class T>
__global__ void add_elements_kernel(const T *x, const T *y, T *result, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  result[gid] = x[gid] + y[gid];
}

template <class T> int vec_add(const T *x, const T *y, T *result, const unsigned count) {
  add_elements_kernel<T><<<(count - 1) / 32 + 1, 32>>>(x, y, result, count);
  int error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

template <class T>
__global__ void sub_elements_kernel(const T *x, const T *y, T *result, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  result[gid] = x[gid] - y[gid];
}

template <class T> int vec_sub(const T *x, const T *y, T *result, const unsigned count) {
  sub_elements_kernel<T><<<(count - 1) / 32 + 1, 32>>>(x, y, result, count);
  int error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}
