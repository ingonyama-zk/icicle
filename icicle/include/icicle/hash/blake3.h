#pragma once

#include "icicle/hash/hash.h"

namespace icicle {

  /**
   * @brief Creates a Blake3 hash object.
   *
   * This function constructs a Hash object configured for Blake3, with the
   * appropriate backend selected based on the current device.
   *
   * @param input_chunk_size size of input in bytes for the Blake3 hash.
   * @return Hash object encapsulating the Blake3 backend.
   */
  Hash create_blake3_hash(uint64_t input_chunk_size = 0);
  struct Blake3 {
    inline static Hash create(uint64_t input_chunk_size = 0) { return create_blake3_hash(input_chunk_size); }
  };
} // namespace icicle