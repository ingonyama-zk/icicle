#include <cstdint>
#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"

#include "hash/hash.cuh"
#include "hash/keccak/keccak.cuh"
#include "kernels.cu"

using namespace hash;

namespace keccak {
  template <int C, int D>
  cudaError_t
  keccak_hash(uint8_t* input, int input_block_size, int number_of_blocks, uint8_t* output, KeccakConfig& config)
  {
    CHK_INIT_IF_RETURN();
    cudaStream_t& stream = config.ctx.stream;

    uint8_t* input_device;
    if (config.are_inputs_on_device) {
      input_device = input;
    } else {
      CHK_IF_RETURN(cudaMallocAsync(&input_device, number_of_blocks * input_block_size, stream));
      CHK_IF_RETURN(
        cudaMemcpyAsync(input_device, input, number_of_blocks * input_block_size, cudaMemcpyHostToDevice, stream));
    }

    uint8_t* output_device;
    if (config.are_outputs_on_device) {
      output_device = output;
    } else {
      CHK_IF_RETURN(cudaMallocAsync(&output_device, number_of_blocks * (D / 8), stream));
    }

    int number_of_threads = 512;
    int number_of_gpu_blocks = (number_of_blocks - 1) / number_of_threads + 1;
    keccak_hash_blocks<C, D><<<number_of_gpu_blocks, number_of_threads, 0, stream>>>(
      input_device, input_block_size, number_of_blocks, output_device);

    if (!config.are_inputs_on_device) CHK_IF_RETURN(cudaFreeAsync(input_device, stream));

    if (!config.are_outputs_on_device) {
      CHK_IF_RETURN(cudaMemcpyAsync(output, output_device, number_of_blocks * (D / 8), cudaMemcpyDeviceToHost, stream));
      CHK_IF_RETURN(cudaFreeAsync(output_device, stream));
    }

    if (!config.is_async) return CHK_STICKY(cudaStreamSynchronize(stream));
    return CHK_LAST();
  }
} // namespace keccak