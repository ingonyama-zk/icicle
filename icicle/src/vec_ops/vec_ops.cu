#include <cuda.h>
#include <stdexcept>

#include "vec_ops/vec_ops.cuh"
#include "gpu-utils/device_context.cuh"
#include "utils/mont.cuh"

namespace vec_ops {

  namespace {

#define MAX_THREADS_PER_BLOCK 256

    template <typename E>
    __global__ void mul_kernel(const E* scalar_vec, const E* element_vec, int n, E* result)
    {
      int tid = blockDim.x * blockIdx.x + threadIdx.x;
      if (tid < n) { result[tid] = scalar_vec[tid] * element_vec[tid]; }
    }

    template <typename E, typename S>
    __global__ void mul_scalar_kernel(const E* element_vec, const S scalar, int n, E* result)
    {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < n) { result[tid] = element_vec[tid] * (scalar); }
    }

    template <typename E>
    __global__ void div_element_wise_kernel(const E* element_vec1, const E* element_vec2, int n, E* result)
    {
      // TODO:implement better based on https://eprint.iacr.org/2008/199
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < n) { result[tid] = element_vec1[tid] * E::inverse(element_vec2[tid]); }
    }

    template <typename E>
    __global__ void add_kernel(const E* element_vec1, const E* element_vec2, int n, E* result)
    {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < n) { result[tid] = element_vec1[tid] + element_vec2[tid]; }
    }

    template <typename E>
    __global__ void sub_kernel(const E* element_vec1, const E* element_vec2, int n, E* result)
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

    template <typename E>
    __global__ void bit_reverse_kernel(const E* input, unsigned n, unsigned shift, E* output)
    {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      // Handling arbitrary vector size
      if (tid < n) {
        int reversed_index = __brev(tid) >> shift;
        output[reversed_index] = input[tid];
      }
    }
    template <typename E>
    __global__ void bit_reverse_inplace_kernel(E* input, unsigned n, unsigned shift)
    {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      // Handling arbitrary vector size
      if (tid < n) {
        int reversed_index = __brev(tid) >> shift;
        if (reversed_index > tid) {
          E temp = input[tid];
          input[tid] = input[reversed_index];
          input[reversed_index] = temp;
        }
      }
    }
  } // namespace

  template <typename E, void (*Kernel)(const E*, const E*, int, E*)>
  cudaError_t vec_op(const E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    CHK_INIT_IF_RETURN();

    // Set the grid and block dimensions
    int num_threads = MAX_THREADS_PER_BLOCK;
    int num_blocks = (n + num_threads - 1) / num_threads;

    E *d_result, *d_alloc_vec_a, *d_alloc_vec_b;
    const E *d_vec_a, *d_vec_b;
    if (!config.is_a_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_alloc_vec_a, n * sizeof(E), config.ctx.stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_alloc_vec_a, vec_a, n * sizeof(E), cudaMemcpyHostToDevice, config.ctx.stream));
      d_vec_a = d_alloc_vec_a;
    } else {
      d_vec_a = vec_a;
    }

    if (!config.is_b_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_alloc_vec_b, n * sizeof(E), config.ctx.stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_alloc_vec_b, vec_b, n * sizeof(E), cudaMemcpyHostToDevice, config.ctx.stream));
      d_vec_b = d_alloc_vec_b;
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

    if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync(d_alloc_vec_a, config.ctx.stream)); }
    if (!config.is_b_on_device) { CHK_IF_RETURN(cudaFreeAsync(d_alloc_vec_b, config.ctx.stream)); }

    if (!config.is_result_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, n * sizeof(E), cudaMemcpyDeviceToHost, config.ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_result, config.ctx.stream));
    }

    if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(config.ctx.stream));

    return CHK_LAST();
  }

  template <typename E>
  cudaError_t mul(const E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    return vec_op<E, mul_kernel>(vec_a, vec_b, n, config, result);
  }

  template <typename E>
  cudaError_t add(const E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    return vec_op<E, add_kernel>(vec_a, vec_b, n, config, result);
  }

  template <typename E>
  cudaError_t sub(const E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    return vec_op<E, sub_kernel>(vec_a, vec_b, n, config, result);
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

  template <typename E>
  cudaError_t bit_reverse(const E* input, unsigned size, BitReverseConfig& cfg, E* output)
  {
    if (size & (size - 1)) THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "bit_reverse: size must be a power of 2");
    if ((input == output) & (cfg.is_input_on_device != cfg.is_output_on_device))
      THROW_ICICLE_ERR(
        IcicleError_t::InvalidArgument, "bit_reverse: equal devices should have same is_on_device parameters");

    E* d_output;
    if (cfg.is_output_on_device) {
      d_output = output;
    } else {
      // allocate output on gpu
      CHK_IF_RETURN(cudaMallocAsync(&d_output, sizeof(E) * size, cfg.ctx.stream));
    }

    unsigned shift = __builtin_clz(size) + 1;
    unsigned num_blocks = (size + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

    if ((input != output) & cfg.is_input_on_device) {
      bit_reverse_kernel<<<num_blocks, MAX_THREADS_PER_BLOCK, 0, cfg.ctx.stream>>>(input, size, shift, d_output);
    } else {
      if (!cfg.is_input_on_device) {
        CHK_IF_RETURN(cudaMemcpyAsync(d_output, input, sizeof(E) * size, cudaMemcpyHostToDevice, cfg.ctx.stream));
      }
      bit_reverse_inplace_kernel<<<num_blocks, MAX_THREADS_PER_BLOCK, 0, cfg.ctx.stream>>>(d_output, size, shift);
    }
    if (!cfg.is_output_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(output, d_output, sizeof(E) * size, cudaMemcpyDeviceToHost, cfg.ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_output, cfg.ctx.stream));
    }
    if (!cfg.is_async) CHK_IF_RETURN(cudaStreamSynchronize(cfg.ctx.stream));
    return CHK_LAST();
  }
} // namespace vec_ops
