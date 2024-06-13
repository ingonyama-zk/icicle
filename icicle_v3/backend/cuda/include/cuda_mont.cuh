#include <cuda.h>
#include <stdexcept>

#include "icicle/errors.h"
#include "icicle/vec_ops.h"
#include "gpu-utils/error_handler.h"

namespace montgomery {
#define MAX_THREADS_PER_BLOCK 256

  template <typename E, bool is_into>
  __global__ void MontgomeryKernel(const E* input, int n, E* output)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) { output[tid] = is_into ? E::to_montgomery(input[tid]) : E::from_montgomery(input[tid]); }
  }

  template <typename E, bool is_into>
  cudaError_t ConvertMontgomery(const E* input, size_t n, const VecOpsConfig& config, E* output)
  {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(config.stream);

    E *d_alloc_out = nullptr, *d_alloc_in = nullptr, *d_out;
    const E* d_in;
    if (!config.is_a_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_alloc_in, n * sizeof(E), cuda_stream));
      CHK_IF_RETURN(cudaMemcpyAsync(d_alloc_in, input, n * sizeof(E), cudaMemcpyHostToDevice, cuda_stream));
      d_in = d_alloc_in;
    } else {
      d_in = input;
    }

    if (!config.is_result_on_device) {
      CHK_IF_RETURN(cudaMallocAsync(&d_alloc_out, n * sizeof(E), cuda_stream));
      d_out = d_alloc_out;
    } else {
      d_out = output;
    }

    int num_threads = MAX_THREADS_PER_BLOCK;
    int num_blocks = (n + num_threads - 1) / num_threads;
    MontgomeryKernel<E, is_into><<<num_blocks, num_threads, 0, cuda_stream>>>(d_in, n, d_out);

    if (d_alloc_in) { CHK_IF_RETURN(cudaFreeAsync(d_alloc_in, cuda_stream)); }
    if (d_alloc_out) {
      CHK_IF_RETURN(cudaMemcpyAsync(output, d_out, n * sizeof(E), cudaMemcpyDeviceToHost, cuda_stream));
      CHK_IF_RETURN(cudaFreeAsync(d_out, cuda_stream));
    }
    if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(cuda_stream));

    return CHK_LAST();
  }

} // namespace montgomery