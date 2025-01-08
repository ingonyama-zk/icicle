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

    constexpr int SIZE_OF_BITS = sizeof(int) * 8 - 1;

    template <typename E, typename S>
    __global__ void fold_kernel(S* values, const E* factors, const int level, const int nlog2, const int n, E* result)
    {
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      if (idx < n) {
        // TODO: this kernel is very basic - improve with shared mem and same
        //       optimizations as ntt, also result on first step can be n/2

        // Step size doubles at each level
        int step = 1 << (level + 1);
        int invlevel = nlog2 - level - 1;

        if (idx < n / step) {
          // Compute indices for the pair to be reduced
          int leftIdx = idx * step;
          int rightIdx = leftIdx + step / 2;

          // Perform the folding operation
          result[leftIdx] = values[leftIdx] + factors[invlevel] * values[rightIdx];
        }
      }
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
  cudaError_t mul(E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    return vec_op<E, mul_kernel>(vec_a, vec_b, n, config, result);
  }

  template <typename E>
  cudaError_t add(E* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  {
    return vec_op<E, add_kernel>(vec_a, vec_b, n, config, result);
  }

  template <typename E, typename S>
  cudaError_t fold(S* vec_a, const E* vec_b, int n, VecOpsConfig& config, E* result)
  // TODO: asap modify vec_op template so it can accomodate minor flexibility without spreading the copypaste
  {
    CHK_INIT_IF_RETURN();

    int nlog2 = log2(n);

    // Set the grid and block dimensions
    int num_threads = 1024;
    int num_blocks = (n + num_threads - 1) / num_threads;

    E* d_result;
    E* d_tmp_result;
    CHK_IF_RETURN(cudaMallocAsync(&d_tmp_result, n * sizeof(E), config.ctx.stream));

    S *d_vec_a, *d_alloc_vec_a;
    E* d_alloc_vec_b;
    const E* d_vec_b;
    if (!config.is_a_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_alloc_vec_a, n * sizeof(S), config.ctx.stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_alloc_vec_a, vec_a, n * sizeof(S), cudaMemcpyHostToDevice, config.ctx.stream));
      d_vec_a = d_alloc_vec_a;
    } else {
      d_vec_a = vec_a;
    }

    if (!config.is_b_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_alloc_vec_b, nlog2 * sizeof(E), config.ctx.stream));
      CHK_IF_RETURN(
        cudaMemcpyAsync(d_alloc_vec_b, vec_b, nlog2 * sizeof(E), cudaMemcpyHostToDevice, config.ctx.stream));
      d_vec_b = d_alloc_vec_b;
    } else {
      d_vec_b = vec_b;
    }

    if (!config.is_result_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_result, sizeof(E), config.ctx.stream));
    } else {
      d_result = result;
    }

    // Call the kernel to perform element-wise operation

    // auto start = std::chrono::high_resolution_clock::now();

    fold_kernel<<<num_blocks, num_threads, 0, config.ctx.stream>>>(d_vec_a, d_vec_b, 0, nlog2, n, d_tmp_result);

    for (int level = 1; level < nlog2; ++level) {
      fold_kernel<<<num_blocks, num_threads, 0, config.ctx.stream>>>(
        d_tmp_result, d_vec_b, level, nlog2, n, d_tmp_result);
    }

    // auto end = std::chrono::high_resolution_clock::now();

    // std::chrono::duration<double, std::milli> elapsed = end - start;
    // std::cout << "time for 2^" << nlog2 << " is: " << elapsed.count() << " ms" << std::endl;

    CHK_IF_RETURN(cudaMemcpyAsync(d_result, d_tmp_result, sizeof(E), cudaMemcpyDeviceToDevice, config.ctx.stream));
    CHK_IF_RETURN(cudaFreeAsync(d_tmp_result, config.ctx.stream));

    if (!config.is_result_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(result, d_result, (size_t)sizeof(E), cudaMemcpyDeviceToHost, config.ctx.stream));
      CHK_IF_RETURN(cudaFreeAsync(d_result, config.ctx.stream));
    }

    if (!config.is_a_on_device) { CHK_IF_RETURN(cudaFreeAsync(d_alloc_vec_a, config.ctx.stream)); }
    if (!config.is_b_on_device) { CHK_IF_RETURN(cudaFreeAsync(d_alloc_vec_b, config.ctx.stream)); }

    if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(config.ctx.stream));

    return CHK_LAST();
  }

  template <typename E>
  cudaError_t
  stwo_convert(uint32_t* vec_a, uint32_t* vec_b, uint32_t* vec_c, uint32_t* vec_d, int n, E* result, bool is_async)
  {
    CHK_INIT_IF_RETURN();
    device_context::DeviceContext ctx = device_context::get_default_device_context();

    uint32_t* d_allocated_input = nullptr;
    uint32_t* d_allocated_output = (uint32_t*)result;

    int size_n = n * sizeof(uint32_t);
    cudaStream_t stream1, stream2, stream3, stream4;
    CHK_IF_RETURN(cudaStreamCreate(&stream1));
    CHK_IF_RETURN(cudaStreamCreate(&stream2));
    CHK_IF_RETURN(cudaStreamCreate(&stream3));
    CHK_IF_RETURN(cudaStreamCreate(&stream4));

    CHK_IF_RETURN(cudaMallocAsync(&d_allocated_input, 4 * size_n, stream1));
    // CHK_IF_RETURN(cudaStreamSynchronize(stream1));

    CHK_IF_RETURN(cudaMemcpyAsync(d_allocated_input, vec_a, size_n, cudaMemcpyHostToDevice, stream1));
    CHK_IF_RETURN(cudaMemcpyAsync(d_allocated_input + n, vec_b, size_n, cudaMemcpyHostToDevice, stream2));
    CHK_IF_RETURN(cudaMemcpyAsync(d_allocated_input + 2 * n, vec_c, size_n, cudaMemcpyHostToDevice, stream3));
    CHK_IF_RETURN(cudaMemcpyAsync(d_allocated_input + 3 * n, vec_d, size_n, cudaMemcpyHostToDevice, stream4));

    CHK_IF_RETURN(cudaStreamSynchronize(stream1));
    CHK_IF_RETURN(cudaStreamSynchronize(stream2));
    CHK_IF_RETURN(cudaStreamSynchronize(stream3));
    CHK_IF_RETURN(cudaStreamSynchronize(stream4));

    CHK_IF_RETURN(cudaStreamDestroy(stream1));
    CHK_IF_RETURN(cudaStreamDestroy(stream2));
    CHK_IF_RETURN(cudaStreamDestroy(stream3));
    CHK_IF_RETURN(cudaStreamDestroy(stream4));

    // TODO: transpose is sup ineficient :(
    CHK_IF_RETURN(transpose_matrix<uint32_t>(d_allocated_input, d_allocated_output, n, 4, ctx, true, true));

    CHK_IF_RETURN(cudaFreeAsync(d_allocated_input, ctx.stream));
    if (!is_async) return CHK_STICKY(cudaStreamSynchronize(ctx.stream));

    return CHK_LAST();
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
