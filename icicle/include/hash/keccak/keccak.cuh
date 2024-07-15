#pragma once
#ifndef KECCAK_H
#define KECCAK_H

#include <cstdint>
#include "gpu-utils/device_context.cuh"
#include "gpu-utils/error_handler.cuh"

#include "hash/hash.cuh"

using namespace hash;

namespace keccak {
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

    Keccak(unsigned int rate) : Hasher<uint8_t, uint64_t>(25, 25, rate, 0) {}
  };
} // namespace keccak

#endif