#pragma once

#include "icicle/hash/hash.h"

namespace icicle {

  /**
   * @brief SumcheckTranscriptConfig describes how prover messages are encoded and hashed
   *        to generate randomness in the Sumcheck protocol.
   *
   * The configuration defines:
   * - Labels for various protocol stages (e.g., domain separator, round polynomials, challenges).
   * - Endianness for encoding messages (little-endian by default).
   * - The hashing function used to derive randomness deterministically.
   * - A seed for initializing the RNG.
   *
   * @tparam S The type for the RNG seed. It should be convenient for both users and FFI integration.
   */
  template <typename S>
  struct SumcheckTranscriptConfig {
    Hash hasher; ///< Hash function used for randomness generation.
    // TODO: Should labels be user-configurable or hardcoded?
    const char* domain_separator_label; ///< Label for the domain separator in the transcript.
    const char* round_poly_label;       ///< Label for round polynomials in the transcript.
    const char* round_challenge_label;  ///< Label for round challenges in the transcript.
    const bool little_endian = true;    ///< Encoding endianness (default: little-endian).
    S seed_rng;                         ///< Seed for initializing the RNG.
  };

  /*
   * Notes:
   * 1. Both Prover and Verifier use this configuration to ensure deterministic randomness generation.
   * 2. Messages (e.g., field elements, strings) are encoded in little/big-endian format based on
   `SumcheckTranscriptConfig.little_endian`.

   *
   * Protocol Workflow:
   *
   * (1) Initialize Transcript:
   * - entry_DS = [domain_separator_label || proof.num_vars || proof.degree || public (hardcoded?) || claimed_sum]
   *
   * (2) First Round:
   * - entry_0 = [round_poly_label || r_0[x].len() || k=0 || r_0[x]]
   * - alpha_0 = Hash(entry_DS || seed_rng || round_challenge_label || entry_0).to_field()
   *
   * (3) Round i > 0:
   * - entry_i = [round_poly_label || r_i[x].len() || k=i || r_i[x]]
   * - alpha_i = Hash(entry_0 || alpha_(i-1) || round_challenge_label || entry_i).to_field()
   *
   */

} // namespace icicle