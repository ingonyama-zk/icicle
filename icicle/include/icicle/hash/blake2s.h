#pragma once

#include "icicle/hash/hash.h"

namespace icicle {

  /**
   * @brief Creates a Keccak-256 hash object.
   *
   * This function constructs a Hash object configured for Keccak-256, with the
   * appropriate backend selected based on the current device.
   *
   * @param total_input_limbs Number of input limbs for the Keccak-256 hash.
   * @return Hash object encapsulating the Keccak-256 backend.
   */
  Hash create_blake2s_hash(uint64_t total_input_limbs);



} // namespace icicle