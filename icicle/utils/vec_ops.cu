#include <cuda.h>
#include <stdexcept>

#include "vec_ops.cuh"
#include "../curves/curve_config.cuh"
#include "device_context.cuh"
#include "mont.cuh"
#include "utils/utils.h"

namespace vec_ops {

  namespace {

#define MAX_THREADS_PER_BLOCK 256

    template <typename E>
    __global__ void MulKernel(E* scalar_vec, E* element_vec, int n, E* result)
    {
      int tid = blockDim.x * blockIdx.x + threadIdx.x;
      if (tid < n) { result[tid] = scalar_vec[tid] * element_vec[tid]; }
    }

    template <typename E>
    __global__ void AddKernel(E* element_vec1, E* element_vec2, int n, E* result)
    {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < n) { result[tid] = element_vec1[tid] + element_vec2[tid]; }
    }

    template <typename E>
    __global__ void SubKernel(E* element_vec1, E* element_vec2, int n, E* result)
    {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < n) { result[tid] = element_vec1[tid] - element_vec2[tid]; }
    }
  } // namespace

  template <typename E, void (*Kernel)(E*, E*, int, E*)>
  cudaError_t VecOp(E* vec_a, E* vec_b, int n, VecOpsConfig<E>& config, E* result)
  {
    CHK_INIT_IF_RETURN();

    // Set the grid and block dimensions
    int num_threads = MAX_THREADS_PER_BLOCK;
    int num_blocks = (n + num_threads - 1) / num_threads;

    E *d_vec_a, *d_vec_b, *d_result;
    if (!config.is_a_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_vec_a, n * sizeof(E), config.ctx.stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_vec_a, vec_a, n * sizeof(E), cudaMemcpyHostToDevice, config.ctx.stream));
    } else {
      d_vec_a = vec_a;
    }

    if (!config.is_b_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_vec_b, n * sizeof(E), config.ctx.stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_vec_b, vec_b, n * sizeof(E), cudaMemcpyHostToDevice, config.ctx.stream));
    } else {
      d_vec_b = vec_b;
    }

    if (!config.is_result_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_result, n * sizeof(E), config.ctx.stream));
    } else {
      d_result = result;
    }

    // Call the kernel to perform element-wise operation
    Kernel<<<num_blocks, num_threads, 0, config.ctx.stream>>>(d_vec_a, d_vec_b, n, d_result);
    if (config.is_result_montgomery_form) CHK_IF_RETURN(mont::FromMontgomery(d_result, n, config.ctx.stream, d_result));

    if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync(d_vec_a, config.ctx.stream)); }

    if (!config.is_b_on_device) { CHK_IF_RETURN(cudaFreeAsync(d_vec_b, config.ctx.stream)); }

    if (!config.is_result_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, n * sizeof(E), cudaMemcpyDeviceToHost, config.ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_result, config.ctx.stream));
    }

    if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(config.ctx.stream));

    return CHK_LAST();
  }

  template <typename E>
  cudaError_t Mul(E* vec_a, E* vec_b, int n, VecOpsConfig<E>& config, E* result)
  {
    return VecOp<E, MulKernel>(vec_a, vec_b, n, config, result);
  }

  template <typename E>
  cudaError_t Add(E* vec_a, E* vec_b, int n, VecOpsConfig<E>& config, E* result)
  {
    return VecOp<E, AddKernel>(vec_a, vec_b, n, config, result);
  }

  template <typename E>
  cudaError_t Sub(E* vec_a, E* vec_b, int n, VecOpsConfig<E>& config, E* result)
  {
    return VecOp<E, SubKernel>(vec_a, vec_b, n, config, result);
  }

  /**
   * Extern version of [Mul](@ref Mul) function with the template parameters
   * `S` and `E` being the [scalar field](@ref scalar_t) of the curve given by `-DCURVE` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, MulCuda)(
    curve_config::scalar_t* vec_a,
    curve_config::scalar_t* vec_b,
    int n,
    VecOpsConfig<curve_config::scalar_t>& config,
    curve_config::scalar_t* result)
  {
    return Mul<curve_config::scalar_t>(vec_a, vec_b, n, config, result);
  }

  /**
   * Extern version of [Add](@ref Add) function with the template parameter
   * `E` being the [scalar field](@ref scalar_t) of the curve given by `-DCURVE` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, AddCuda)(
    curve_config::scalar_t* vec_a,
    curve_config::scalar_t* vec_b,
    int n,
    VecOpsConfig<curve_config::scalar_t>& config,
    curve_config::scalar_t* result)
  {
    return Add<curve_config::scalar_t>(vec_a, vec_b, n, config, result);
  }

  /**
   * Extern version of [Sub](@ref Sub) function with the template parameter
   * `E` being the [scalar field](@ref scalar_t) of the curve given by `-DCURVE` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, SubCuda)(
    curve_config::scalar_t* vec_a,
    curve_config::scalar_t* vec_b,
    int n,
    VecOpsConfig<curve_config::scalar_t>& config,
    curve_config::scalar_t* result)
  {
    return Sub<curve_config::scalar_t>(vec_a, vec_b, n, config, result);
  }

} // namespace vec_ops