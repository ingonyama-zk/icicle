#pragma once
#ifndef KECCAK_H
#define KECCAK_H

#include <cstdint>
#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"

#include "hash/hash.cuh"

using namespace hash;

namespace keccak {
  // Input rate in bytes
  const int KECCAK_256_RATE = 136;
  const int KECCAK_512_RATE = 72;

  // Digest size in u64
  const int KECCAK_256_DIGEST = 4;
  const int KECCAK_512_DIGEST = 8;

  // Number of state elements in u64
  const int KECCAK_STATE_SIZE = 25;

  class Keccak : public Hasher<uint8_t, uint64_t>
  {
  public:
    cudaError_t run_hash_many_kernel(
      const uint8_t* input,
      uint64_t* output,
      unsigned int number_of_states,
      unsigned int input_len,
      unsigned int output_len,
      const device_context::DeviceContext& ctx) const override;

    Keccak(unsigned int rate) : Hasher<uint8_t, uint64_t>(KECCAK_STATE_SIZE, KECCAK_STATE_SIZE, rate, 0) {}
  };
} // namespace keccak

#endif