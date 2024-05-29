#pragma once
#ifndef KECCAK_H
#define KECCAK_H

#include <cstdint>
#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"

#include "hash/hash.cuh"

using namespace hash;

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

  class Keccak : public Hash<uint64_t>, public SpongeHasher<Keccak, uint8_t, uint64_t>, public CompressionHasher<Keccak, uint64_t>
  {
  public:
    static const int KECCAK_BLOCK_SIZE = 128;

    static inline int keccak_number_of_blocks(size_t number_of_states)
    {
      return (number_of_states - 1) / KECCAK_BLOCK_SIZE + 1;
    }

    cudaError_t pad_many(
      uint64_t* states,
      unsigned int number_of_states,
      unsigned int input_block_len,
      unsigned int rate,
      device_context::DeviceContext& ctx) const override;

    cudaError_t squeeze_states(
      const uint64_t* states,
      unsigned int number_of_states,
      unsigned int rate,
      unsigned int offset,
      bool align,
      uint64_t* output,
      device_context::DeviceContext& ctx) const override;

    cudaError_t run_permutation_kernel(
      const uint64_t* states,
      uint64_t* output,
      unsigned int number_of_states,
      bool aligned,
      device_context::DeviceContext& ctx) const override;

    Keccak() : Hash<uint64_t>(25, 25){};
    ~Keccak() = default;
  };
} // namespace keccak

#endif