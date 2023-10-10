#include "utils_kernels.cuh"

namespace utils_internal {

// template <typename E, typename S>
// __global__ void template_normalize_kernel(E* arr, S scalar, int n)
// {
//   int tid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (tid < n) { arr[tid] = scalar * arr[tid]; }
// }

// template <typename E, typename S>
// __global__ void batchVectorMult(E* element_vec, S* scalar_vec, unsigned n_scalars, unsigned batch_size)
// {
//   int tid = blockDim.x * blockIdx.x + threadIdx.x;
//   if (tid < n_scalars * batch_size) {
//     int scalar_id = tid % n_scalars;
//     element_vec[tid] = scalar_vec[scalar_id] * element_vec[tid];
//   }
// }

} // namespace utils_internal
