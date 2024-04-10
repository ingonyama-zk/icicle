#include <cuda.h>
#include <stdexcept>

#include "vec_ops/vec_ops.cuh"
#include "gpu-utils/device_context.cuh"
#include "utils/mont.cuh"

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

    template <typename E>
    __global__ void transpose_kernel(const E* in, E* out, uint32_t row_size, uint32_t column_size)
    {
      int tid = blockDim.x * blockIdx.x + threadIdx.x;
      if (tid >= row_size * column_size) return;
      out[(tid % row_size) * column_size + (tid / row_size)] = in[tid];
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

  template <typename E>
  cudaError_t transpose_matrix(
    const E* mat_in,
    E* mat_out,
    uint32_t row_size,
    uint32_t column_size,
    device_context::DeviceContext& ctx,
    bool on_device,
    bool is_async)
  {
    int number_of_threads = MAX_THREADS_PER_BLOCK;
    int number_of_blocks = (row_size * column_size + number_of_threads - 1) / number_of_threads;
    cudaStream_t stream = ctx.stream;

    const E* d_mat_in;
    E* d_allocated_input = nullptr;
    E* d_mat_out;
    if (!on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_allocated_input, row_size * column_size * sizeof(E), ctx.stream));
      CHK_IF_RETURN(cudaMemcpyAsync(
        d_allocated_input, mat_in, row_size * column_size * sizeof(E), cudaMemcpyHostToDevice, ctx.stream));

      CHK_IF_RETURN(cudaMallocAsync(&d_mat_out, row_size * column_size * sizeof(E), ctx.stream));
      d_mat_in = d_allocated_input;
    } else {
      d_mat_in = mat_in;
      d_mat_out = mat_out;
    }

    transpose_kernel<<<number_of_blocks, number_of_threads, 0, stream>>>(d_mat_in, d_mat_out, row_size, column_size);

    if (!on_device) {
      CHK_IF_RETURN(
        cudaMemcpyAsync(mat_out, d_mat_out, row_size * column_size * sizeof(E), cudaMemcpyDeviceToHost, ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_mat_out, ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_allocated_input, ctx.stream));
    }
    if (!is_async) return CHK_STICKY(cudaStreamSynchronize(ctx.stream));

    return CHK_LAST();
  }
} // namespace vec_ops