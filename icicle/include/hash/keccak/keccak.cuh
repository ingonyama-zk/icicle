#pragma once
#ifndef KECCAK_H
#define KECCAK_H

#include <cstdint>
#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"

#include "kernels.cuh"

namespace keccak {
  /**
   * @struct KeccakConfig
   * Struct that encodes various Keccak parameters.
   */
  struct KeccakConfig {
    device_context::DeviceContext ctx; /**< Details related to the device such as its id and stream id. */
    bool are_inputs_on_device;  /**< True if inputs are on device and false if they're on host. Default value: false. */
    bool are_outputs_on_device; /**< If true, output is preserved on device, otherwise on host. Default value: false. */
    bool is_async; /**< Whether to run the Keccak asynchronously. If set to `true`, the keccak_hash function will be
                    *   non-blocking and you'd need to synchronize it explicitly by running
                    *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, keccak_hash
                    *   function will block the current CPU thread. */
  };

  KeccakConfig default_keccak_config()
  {
    device_context::DeviceContext ctx = device_context::get_default_device_context();
    KeccakConfig config = {
      ctx,   // ctx
      false, // are_inputes_on_device
      false, // are_outputs_on_device
      false, // is_async
    };
    return config;
  }

  /**
   * Compute the keccak hash over a sequence of preimages.
   * Takes {number_of_blocks * input_block_size} u64s of input and computes {number_of_blocks} outputs, each of size {D
   * / 64} u64
   * @tparam C - number of bits of capacity (c = b - r = 1600 - r). Only multiples of 64 are supported.
   * @tparam D - number of bits of output. Only multiples of 64 are supported.
   * @param input a pointer to the input data. May be allocated on device or on host, regulated
   * by the config. Must be of size [input_block_size](@ref input_block_size) * [number_of_blocks](@ref
   * number_of_blocks)}.
   * @param input_block_size - size of each input block in bytes. Should be divisible by 8.
   * @param number_of_blocks number of input and output blocks. One GPU thread processes one block
   * @param output a pointer to the output data. May be allocated on device or on host, regulated
   * by the config. Must be of size [output_block_size](@ref output_block_size) * [number_of_blocks](@ref
   * number_of_blocks)}
   */
  template <int C, int D>
  cudaError_t
  keccak_hash(uint8_t* input, int input_block_size, int number_of_blocks, uint8_t* output, KeccakConfig& config);

  template <int C, int D>
  class Keccak : public Permutation<uint_64_t>,
                 public CompressionHasher<uint64_t>,
                 public SpongeHasher<uint8_t, uint64_t>
  {
    const int KECCAK_BLOCK_SIZE = 512;

    cudaError_t squeeze_states(
      const S* states,
      unsigned int number_of_states,
      unsigned int rate,
      S* output,
      device_context::DeviceContext& ctx,
      unsigned int offset = 0) const override
    {
      squeeze_states_kernel<S, width, 1, 0>
        <<<poseidon_number_of_blocks(number_of_states), KECCAK_BLOCK_SIZE, 0, ctx.stream>>>(
          states, number_of_states, output);

      CHK_IF_RETURN(cudaPeekAtLastError());
      return CHK_LAST();
    }

    cudaError_t run_permutation_kernel(
      const S* states, S* output, unsigned int number_of_states, device_context::DeviceContext& ctx) const override
    {
      keccak_permutation_kernel<C, D>
        <<<poseidon_number_of_blocks(number_of_states), KECCAK_BLOCK_SIZE, 0, ctx.stream>>>(
          states, output, number_of_states, this->constants);

      CHK_IF_RETURN(cudaPeekAtLastError());
      return CHK_LAST();
    }
  };

  using Keccak256 = Keccak<512, 256>;
  using Keccak512 = Keccak<1024, 512>;
} // namespace keccak

#endif