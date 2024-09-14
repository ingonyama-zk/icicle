#include <cuda.h>
#include <stdexcept>

#include "vec_ops/vec_ops.cuh"
#include "gpu-utils/device_context.cuh"
#include "utils/mont.cuh"
#include <cub/cub.cuh>

namespace vec_ops {

  namespace {

#define MAX_THREADS_PER_BLOCK 256

    template <typename E>
    __global__ void sum_kernel(E* R, const E* A, const E* B, int n)
    {
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      if (idx < n) { R[idx] = A[idx] + B[idx]; }
    }

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
    __global__ void bit_reverse_kernel(const E* input, uint64_t n, unsigned shift, E* output)
    {
      uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
      // Handling arbitrary vector size
      if (tid < n) {
        int reversed_index = __brevll(tid) >> shift;
        output[reversed_index] = input[tid];
      }
    }
    template <typename E>
    __global__ void bit_reverse_inplace_kernel(E* input, uint64_t n, unsigned shift)
    {
      uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
      // Handling arbitrary vector size
      if (tid < n) {
        int reversed_index = __brevll(tid) >> shift;
        if (reversed_index > tid) {
          E temp = input[tid];
          input[tid] = input[reversed_index];
          input[reversed_index] = temp;
        }
      }
    }
  } // namespace

  template <typename E, void (*Kernel)(const E*, const E*, int, E*)>
  cudaError_t vec_op(E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    CHK_INIT_IF_RETURN();

    bool is_in_place = vec_a == result;

    // Set the grid and block dimensions
    int num_threads = MAX_THREADS_PER_BLOCK;
    int num_blocks = (n + num_threads - 1) / num_threads;

    E *d_result, *d_alloc_vec_a, *d_alloc_vec_b;
    E* d_vec_a;
    const E* d_vec_b;
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
      if (!is_in_place) {
        CHK_IF_RETURN(cudaMallocAsync(&d_result, n * sizeof(E), config.ctx.stream));
      } else {
        d_result = d_vec_a;
      }
    } else {
      if (!is_in_place) {
        d_result = result;
      } else {
        d_result = result = d_vec_a;
      }
    }

    // Call the kernel to perform element-wise operation
    Kernel<<<num_blocks, num_threads, 0, config.ctx.stream>>>(d_vec_a, d_vec_b, n, d_result);

    if (!config.is_result_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, n * sizeof(E), cudaMemcpyDeviceToHost, config.ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_result, config.ctx.stream));
    }

    if (!config.is_a_on_device && !is_in_place) { CHK_IF_RETURN(cudaFreeAsync(d_alloc_vec_a, config.ctx.stream)); }
    if (!config.is_b_on_device) { CHK_IF_RETURN(cudaFreeAsync(d_alloc_vec_b, config.ctx.stream)); }

    if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(config.ctx.stream));

    return CHK_LAST();
  }

  template <typename E>
  __global__ void reduceSumKernel(const E* __restrict__ input, E* __restrict__ output, int n)
  {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Specialize BlockReduce for type E.
    typedef cub::BlockReduce<E, 512> BlockReduceT;

    // Allocate temporary storage in shared memory.
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    E result = E::zero();
    if (tid < n) result = input[tid];

    // Perform block reduction and obtain the result.
    result = BlockReduceT(temp_storage).Sum(result);

    // Write result for this block to global memory.
    if (threadIdx.x == 0) output[blockIdx.x] = result;
  }

  template <typename E>
  cudaError_t sum(E* d_input, int n, E* result)
  {
    int blockSize = 512;
    int numBlocks = (n + blockSize - 1) / blockSize;

    E* d_output;
    cudaError_t err = cudaMalloc(&d_output, numBlocks * sizeof(E));
    if (err != cudaSuccess) return err;

    // Launch the reduction kernel
    reduceSumKernel<<<numBlocks, blockSize, blockSize * sizeof(E)>>>(d_input, d_output, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      cudaFree(d_output);
      return err;
    }

    if (numBlocks > 1) {
      // Recursively reduce the result if multiple blocks
      err = sum(d_output, numBlocks, result);
    } else {
      // Copy the final result back to the host
      err = cudaMemcpy(result, d_output, sizeof(E), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_output);
    return err;
  }


  template <typename E>
  __global__ void eval_cubic_kernel(
      const E* __restrict__ a_low, 
      const E* __restrict__ a_high, 
      const E* __restrict__ b_low, 
      const E* __restrict__ b_high, 
      const E* __restrict__ c_low, 
      const E* __restrict__ c_high, 
      E* __restrict__ e_0,
      E* __restrict__ e_1,
      E* __restrict__ e_2,
      E* __restrict__ e_3,
      int n
    ) {
      unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid >= n) return;

      E a_low_val = a_low[tid];
      E a_high_val = a_high[tid];
      E b_low_val = b_low[tid];
      E b_high_val = b_high[tid];
      E c_low_val = c_low[tid];
      E c_high_val = c_high[tid];

      e_0[tid] = a_low_val * b_low_val * c_low_val;
      e_1[tid] = a_high_val * b_high_val * c_high_val;

      E M_a = a_high_val - a_low_val;
      E M_b = b_high_val - b_low_val;
      E M_c = c_high_val - c_low_val;

      E B_a = a_high_val + M_a;
      E B_b = b_high_val + M_b;
      E B_c = c_high_val + M_c;
      e_2[tid] = B_a * B_b * B_c;

      B_a = B_a + M_a;
      B_b = B_b + M_b;
      B_c = B_c + M_c;
      e_3[tid] = B_a * B_b * B_c;
  }



  template <typename E>
  cudaError_t eval_cubic(E* A, E* B, E* C, int n, E* result)
  {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    E* E_all;
    cudaError_t err = cudaMalloc(&E_all, 4 * n * sizeof(E));
    if (err != cudaSuccess) return err;

    E* E_0 = E_all;
    E* E_1 = E_all + n;
    E* E_2 = E_all + 2 * n;
    E* E_3 = E_all + 3 * n;

    E* a_low = A;
    E* a_high = A + n;
    E* b_low = B;
    E* b_high = B + n;
    E* c_low = C;
    E* c_high = C + n;

    eval_cubic_kernel<<<numBlocks, blockSize>>>(a_low, a_high, b_low, b_high, c_low, c_high, E_0, E_1, E_2, E_3, n);
    E* eval_points;
    err = cudaMalloc(&eval_points, 4 * sizeof(E));
    if (err != cudaSuccess) {
        cudaFree(E_all);
        return err;
    }

    sum(E_0, n, &eval_points[0]);
    sum(E_1, n, &eval_points[1]);
    sum(E_2, n, &eval_points[2]);
    sum(E_3, n, &eval_points[3]);

    CHK_IF_RETURN(cudaMemcpy(result, eval_points, sizeof(E) * 4, cudaMemcpyDeviceToHost));
    CHK_IF_RETURN(cudaFree(E_all));
    CHK_IF_RETURN(cudaFree(eval_points));

    if (err != cudaSuccess) {
      return err;
    }

    return cudaSuccess;
  }

  template <typename E>
  __global__ void bind_kernel(
      E* __restrict__ low, 
      E* __restrict__ high, 
      int n,
      E r
  ) {
      unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid >= n) return;

      low[tid] = low[tid] + (r * (high[tid] - low[tid]));
  }

  template <typename E>
  cudaError_t bind(E* vec, int n, E r)
  {
    int numBlocks = (n + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    int blockSize = MAX_THREADS_PER_BLOCK;

    E* low = vec;
    E* high = vec + n;

    bind_kernel<<<numBlocks, blockSize>>>(low, high, n, r);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return err;
    }

    // Synchronize to ensure all operations are complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return err;
    }

    return cudaSuccess;
  }

  // TODO(sragss): Try bind triple
  template <typename E>
  cudaError_t bind_triple(E* a, E* b, E* c, int n, E r)
  {
    int numBlocks = (n + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    int blockSize = MAX_THREADS_PER_BLOCK;

    bind_kernel<<<numBlocks, blockSize>>>(a, a + n, n, r);
    bind_kernel<<<numBlocks, blockSize>>>(b, b + n, n, r);
    bind_kernel<<<numBlocks, blockSize>>>(c, c + n, n, r);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return err;
    }

    // Synchronize to ensure all operations are complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return err;
    }

    return cudaSuccess;
  }

  template <typename E>
  cudaError_t mul(E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    return vec_op<E, mul_kernel>(vec_a, vec_b, n, config, result);
  }

  template <typename E>
  cudaError_t add(E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    return vec_op<E, add_kernel>(vec_a, vec_b, n, config, result);
  }

  template <typename E>
  cudaError_t sub(E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    return vec_op<E, sub_kernel>(vec_a, vec_b, n, config, result);
  }

  template <typename E>
  cudaError_t transpose_matrix(
    const E* mat_in,
    E* mat_out,
    uint32_t row_size,
    uint32_t column_size,
    const device_context::DeviceContext& ctx,
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
  cudaError_t bit_reverse(const E* input, uint64_t size, BitReverseConfig& cfg, E* output)
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

    uint64_t shift = __builtin_clzll(size) + 1;
    uint64_t num_blocks = (size + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

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
