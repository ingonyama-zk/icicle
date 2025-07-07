#pragma once

#include <vector>
#include <cstddef>              // std::byte
#include <cstring>              // std::memcpy
#include "icicle/hash/keccak.h" // Sha3_256::create()
#include "examples_utils.h"

/**
 * @brief A very small SHA3-256 based random-oracle.
 *
 * The oracle stores an internal byte-state.  On every call to `generate(...)`
 * it hashes [ state || message ] with SHA3-256, returns the digest, and sets
 * `state = digest`.  With the same seed and the same sequence of messages
 * both Prover and Verifier deterministically obtain identical challenges.
 */
class Oracle
{
public:
  Hash hasher_;                  // SHA3-256 hash engine
  std::vector<std::byte> state_; // current transcript state

  // Construct with an initial seed.
  Oracle(const std::byte* seed, size_t seed_len) : hasher_(Sha3_256::create()), state_(seed, seed + seed_len) {}
  // Copy constructor
  Oracle(const Oracle& other) : hasher_(Sha3_256::create()), state_(other.state_) {}

  /**
   * @brief Produce the next challenge.
   * @param msg   Pointer to the message to absorb.
   * @param len   Length of the message in bytes.
   * @return      A vector containing the SHA3-256 digest.
   */
  std::vector<std::byte> generate(const std::byte* msg, size_t len);
};
