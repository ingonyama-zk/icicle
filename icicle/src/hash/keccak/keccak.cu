#include <cstdint>
#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"

#include "hash/hash.cuh"
#include "hash/keccak/keccak.cuh"
#include "kernels.cu"

using namespace hash;

namespace keccak {
  cudaError_t Keccak::pad_many(
    uint64_t* states,
    unsigned int number_of_states,
    unsigned int input_block_len,
    const device_context::DeviceContext& ctx) const
  {
    unsigned int input_len = input_block_len % this->rate;
    keccak_10_1_pad_kernel<<<keccak_number_of_blocks(number_of_states), KECCAK_BLOCK_SIZE, 0, ctx.stream>>>(
      states, input_len, this->rate, number_of_states);

    CHK_IF_RETURN(cudaPeekAtLastError());
    return CHK_LAST();
  }

  cudaError_t Keccak::squeeze_states(
    const uint64_t* states,
    unsigned int number_of_states,
    unsigned int output_len,
    uint64_t* output,
    const device_context::DeviceContext& ctx) const
  {
    switch (this->rate) {
    case 17:
      squeeze_states_kernel<4><<<keccak_number_of_blocks(number_of_states), KECCAK_BLOCK_SIZE, 0, ctx.stream>>>(
        states, number_of_states, output);
      break;
    case 9:
      squeeze_states_kernel<8><<<keccak_number_of_blocks(number_of_states), KECCAK_BLOCK_SIZE, 0, ctx.stream>>>(
        states, number_of_states, output);
      break;
    default:
      THROW_ICICLE_ERR(IcicleError_t::InvalidArgument, "KeccakSqueeze: #rate must be one of [17, 9]");
    }

    CHK_IF_RETURN(cudaPeekAtLastError());
    return CHK_LAST();
  }

  cudaError_t Keccak::run_permutation_kernel(
    const uint64_t* states,
    uint64_t* output,
    unsigned int number_of_states,
    bool aligned,
    const device_context::DeviceContext& ctx) const
  {
    keccak_permutation_kernel<<<keccak_number_of_blocks(number_of_states), KECCAK_BLOCK_SIZE, 0, ctx.stream>>>(
      states, output, number_of_states);

    CHK_IF_RETURN(cudaPeekAtLastError());
    return CHK_LAST();
  }

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