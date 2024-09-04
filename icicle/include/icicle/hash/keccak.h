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
  Hash create_keccak_256_hash(uint64_t total_input_limbs);

  /**
   * @brief Creates a Keccak-512 hash object.
   *
   * This function constructs a Hash object configured for Keccak-512, with the
   * appropriate backend selected based on the current device.
   *
   * @param total_input_limbs Number of input limbs for the Keccak-512 hash.
   * @return Hash object encapsulating the Keccak-512 backend.
   */
  Hash create_keccak_512_hash(uint64_t total_input_limbs);

  /**
   * @brief Creates a SHA3-256 hash object.
   *
   * This function constructs a Hash object configured for SHA3-256, which is a
   * variant of Keccak-256, with the appropriate backend selected based on the current device.
   *
   * @param total_input_limbs Number of input limbs for the SHA3-256 hash.
   * @return Hash object encapsulating the SHA3-256 backend.
   */
  Hash create_sha3_256_hash(uint64_t total_input_limbs);

  /**
   * @brief Creates a SHA3-512 hash object.
   *
   * This function constructs a Hash object configured for SHA3-512, with the
   * appropriate backend selected based on the current device.
   *
   * @param total_input_limbs Number of input limbs for the SHA3-512 hash.
   * @return Hash object encapsulating the SHA3-512 backend.
   */
  Hash create_sha3_512_hash(uint64_t total_input_limbs);

} // namespace icicle