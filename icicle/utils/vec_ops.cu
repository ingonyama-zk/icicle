#include <cuda.h>
#include <stdexcept>

#include "../curves/curve_config.cuh"
#include "device_context.cuh"
#include "mont.cuh"

namespace vec_ops {

  namespace {

#define MAX_THREADS_PER_BLOCK 256

    template <typename E, typename S>
    __global__ void MulKernel(S* scalar_vec, E* element_vec, int n, E* result)
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

  template <typename E, typename S>
  cudaError_t
  Mul(S* vec_a, E* vec_b, int n, bool is_on_device, bool is_montgomery, device_context::DeviceContext ctx, E* result)
  {
    CHK_INIT_IF_RETURN();

    // Set the grid and block dimensions
    int num_threads = MAX_THREADS_PER_BLOCK;
    int num_blocks = (n + num_threads - 1) / num_threads;

    S* d_vec_a;
    E *d_vec_b, *d_result;
    if (!is_on_device) {
      // Allocate memory on the device for the input vectors and the output vector
      CHK_IF_RETURN(cudaMallocAsync(&d_vec_a, n * sizeof(S), ctx.stream));
      CHK_IF_RETURN(cudaMallocAsync(&d_vec_b, n * sizeof(E), ctx.stream));
      CHK_IF_RETURN(cudaMallocAsync(&d_result, n * sizeof(E), ctx.stream));

      // Copy the input vectors and the modulus from the host to the device
      CHK_IF_RETURN(cudaMemcpyAsync(d_vec_a, vec_a, n * sizeof(S), cudaMemcpyHostToDevice, ctx.stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_vec_b, vec_b, n * sizeof(E), cudaMemcpyHostToDevice, ctx.stream));
    }

    E* d_res = is_on_device ? result : d_result;
    // Call the kernel to perform element-wise modular multiplication
    MulKernel<<<num_blocks, num_threads, 0, ctx.stream>>>(
      is_on_device ? vec_a : d_vec_a, is_on_device ? vec_b : d_vec_b, n, d_res);
    if (is_montgomery) CHK_IF_RETURN(mont::FromMontgomery(d_res, n, ctx.stream, d_res));

    if (!is_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, n * sizeof(E), cudaMemcpyDeviceToHost, ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_vec_a, ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_vec_b, ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_result, ctx.stream));
    }

    return CHK_STICKY(cudaStreamSynchronize(ctx.stream));
  }

  template <typename E>
  cudaError_t Add(E* vec_a, E* vec_b, int n, bool is_on_device, device_context::DeviceContext ctx, E* result)
  {
    CHK_INIT_IF_RETURN();

    // Set the grid and block dimensions
    int num_threads = MAX_THREADS_PER_BLOCK;
    int num_blocks = (n + num_threads - 1) / num_threads;

    E *d_vec_a, *d_vec_b, *d_result;
    if (!is_on_device) {
      // Allocate memory on the device for the input vectors and the output vector
      CHK_IF_RETURN(cudaMallocAsync(&d_vec_a, n * sizeof(E), ctx.stream));
      CHK_IF_RETURN(cudaMallocAsync(&d_vec_b, n * sizeof(E), ctx.stream));
      CHK_IF_RETURN(cudaMallocAsync(&d_result, n * sizeof(E), ctx.stream));

      // Copy the input vectors from the host to the device
      CHK_IF_RETURN(cudaMemcpyAsync(d_vec_a, vec_a, n * sizeof(E), cudaMemcpyHostToDevice, ctx.stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_vec_b, vec_b, n * sizeof(E), cudaMemcpyHostToDevice, ctx.stream));
    }

    // Call the kernel to perform element-wise addition
    AddKernel<<<num_blocks, num_threads, 0, ctx.stream>>>(
      is_on_device ? vec_a : d_vec_a, is_on_device ? vec_b : d_vec_b, n, is_on_device ? result : d_result);

    if (!is_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, n * sizeof(E), cudaMemcpyDeviceToHost, ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_vec_a, ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_vec_b, ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_result, ctx.stream));
    }

    return CHK_STICKY(cudaStreamSynchronize(ctx.stream));
  }

  template <typename E>
  cudaError_t Sub(E* vec_a, E* vec_b, int n, bool is_on_device, device_context::DeviceContext ctx, E* result)
  {
    CHK_INIT_IF_RETURN();

    // Set the grid and block dimensions
    int num_threads = MAX_THREADS_PER_BLOCK;
    int num_blocks = (n + num_threads - 1) / num_threads;

    E *d_vec_a, *d_vec_b, *d_result;
    if (!is_on_device) {
      // Allocate memory on the device for the input vectors and the output vector
      CHK_IF_RETURN(cudaMallocAsync(&d_vec_a, n * sizeof(E), ctx.stream));
      CHK_IF_RETURN(cudaMallocAsync(&d_vec_b, n * sizeof(E), ctx.stream));
      CHK_IF_RETURN(cudaMallocAsync(&d_result, n * sizeof(E), ctx.stream));

      // Copy the input vectors from the host to the device
      CHK_IF_RETURN(cudaMemcpyAsync(d_vec_a, vec_a, n * sizeof(E), cudaMemcpyHostToDevice, ctx.stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_vec_b, vec_b, n * sizeof(E), cudaMemcpyHostToDevice, ctx.stream));
    }

    // Call the kernel to perform element-wise subtraction
    SubKernel<<<num_blocks, num_threads, 0, ctx.stream>>>(
      is_on_device ? vec_a : d_vec_a, is_on_device ? vec_b : d_vec_b, n, is_on_device ? result : d_result);

    if (!is_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, n * sizeof(E), cudaMemcpyDeviceToHost, ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_vec_a, ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_vec_b, ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_result, ctx.stream));
    }

    return CHK_STICKY(cudaStreamSynchronize(ctx.stream));
  }

  /**
   * Extern version of [Mul](@ref Mul) function with the template parameters
   * `S` and `E` being the [scalar field](@ref scalar_t) of the curve given by `-DCURVE` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t MulCuda(
    curve_config::scalar_t* vec_a,
    curve_config::scalar_t* vec_b,
    int n,
    bool is_on_device,
    bool is_montgomery,
    device_context::DeviceContext ctx,
    curve_config::scalar_t* result)
  {
    return Mul<curve_config::scalar_t>(vec_a, vec_b, n, is_on_device, is_montgomery, ctx, result);
  }

  /**
   * Extern version of [Add](@ref Add) function with the template parameter
   * `E` being the [scalar field](@ref scalar_t) of the curve given by `-DCURVE` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t AddCuda(
    curve_config::scalar_t* vec_a,
    curve_config::scalar_t* vec_b,
    int n,
    bool is_on_device,
    device_context::DeviceContext ctx,
    curve_config::scalar_t* result)
  {
    return Add<curve_config::scalar_t>(vec_a, vec_b, n, is_on_device, ctx, result);
  }

  /**
   * Extern version of [Sub](@ref Sub) function with the template parameter
   * `E` being the [scalar field](@ref scalar_t) of the curve given by `-DCURVE` env variable during build.
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t SubCuda(
    curve_config::scalar_t* vec_a,
    curve_config::scalar_t* vec_b,
    int n,
    bool is_on_device,
    device_context::DeviceContext ctx,
    curve_config::scalar_t* result)
  {
    return Sub<curve_config::scalar_t>(vec_a, vec_b, n, is_on_device, ctx, result);
  }

} // namespace vec_ops