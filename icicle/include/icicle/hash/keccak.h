#pragma once

#include "icicle/hash/hash.h"

namespace icicle {

  /**
   * @brief Creates a Keccak-256 hash object.
   *
   * This function constructs a Hash object configured for Keccak-256, with the
   * appropriate backend selected based on the current device.
   *
   * @param default_input_chunk_size default size of input in bytes for the Keccak-256 hash.
   * this value is used when the hash function is called with size=0 and in merkle tree.
   * @return Hash object encapsulating the Keccak-256 backend.
   */
  Hash create_keccak_256_hash(uint64_t default_input_chunk_size = 0);
  struct Keccak256 {
    inline static Hash create(uint64_t default_input_chunk_size = 0)
    {
      return create_keccak_256_hash(default_input_chunk_size);
    }
  };

  /**
   * @brief Creates a Keccak-512 hash object.
   *
   * This function constructs a Hash object configured for Keccak-512, with the
   * appropriate backend selected based on the current device.
   *
   * @param input_chunk_size size of input in bytes for the Keccak-512 hash.
   * @return Hash object encapsulating the Keccak-512 backend.
   */
  Hash create_keccak_512_hash(uint64_t input_chunk_size = 0);
  struct Keccak512 {
    inline static Hash create(uint64_t input_chunk_size = 0) { return create_keccak_512_hash(input_chunk_size); }
  };

  /**
   * @brief Creates a SHA3-256 hash object.
   *
   * This function constructs a Hash object configured for SHA3-256, which is a
   * variant of Keccak-256, with the appropriate backend selected based on the current device.
   *
   * @param input_chunk_size size of input in bytes for the SHA3-256 hash.
   * @return Hash object encapsulating the SHA3-256 backend.
   */
  Hash create_sha3_256_hash(uint64_t input_chunk_size = 0);
  struct Sha3_256 {
    inline static Hash create(uint64_t input_chunk_size = 0) { return create_sha3_256_hash(input_chunk_size); }
  };

  /**
   * @brief Creates a SHA3-512 hash object.
   *
   * This function constructs a Hash object configured for SHA3-512, with the
   * appropriate backend selected based on the current device.
   *
   * @param input_chunk_size size of input in bytes for the SHA3-512 hash.
   * @return Hash object encapsulating the SHA3-512 backend.
   */
  Hash create_sha3_512_hash(uint64_t input_chunk_size = 0);
  struct Sha3_512 {
    inline static Hash create(uint64_t input_chunk_size = 0) { return create_sha3_512_hash(input_chunk_size); }
  };

} // namespace icicle