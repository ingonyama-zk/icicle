#include "utils_kernels.cuh"

namespace utils_internal {
  // TODO: weird linking issue - only works in headers
  // template <typename E, typename S>
  // __global__ void NormalizeKernel(E* arr, S scalar, unsigned n)
  // {
  //   int tid = blockIdx.x * blockDim.x + threadIdx.x;
  //   if (tid < n) { arr[tid] = scalar * arr[tid]; }
  // }

  template <typename E, typename S>
  __global__ void NormalizeKernel(E* arr, S scalar, int n)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) { arr[tid] = scalar * arr[tid]; }
  }

  template <typename E, typename S>
  __global__ void BatchMulKernel(E* element_vec, S* scalar_vec, int n_scalars, int batch_size)
  {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n_scalars * batch_size) {
      int scalar_id = tid % n_scalars;
      element_vec[tid] = scalar_vec[scalar_id] * element_vec[tid];
    }
  }

} // namespace utils_internal
