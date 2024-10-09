#pragma once

#include "icicle/hash/hash.h"

namespace icicle {

  /**
   * @brief Creates a Blake2s hash object.
   *
   * This function constructs a Hash object configured for Blake2s, with the
   * appropriate backend selected based on the current device.
   *
   * @param input_chunk_size size of input in bytes for the Blake2s hash.
   * @return Hash object encapsulating the Blake2s backend.
   */
  Hash create_blake2s_hash(uint64_t input_chunk_size = 0);
  struct Blake2s {
    inline static Hash create(uint64_t input_chunk_size = 0) { return create_blake2s_hash(); }
  };
} // namespace icicle