#pragma once

#include "icicle/hash/hash.h"

/* This file defines the SumcheckTranscriptConfig class, which specifies how to encode and hash prover messages
 * in the Sumcheck protocol, ensuring deterministic randomness generation and correct message encoding.
 *
 * Message structure:
 *
 * encoded_round_msg = [label || length(prover_msg) || round_number || prover_msg_in_round]
 * hash(public || prev_challenge || round_challenge_label || encoded_round_msg)
 *
 * Protocol Workflow:
 *
 * 1. Initialize Transcript:
 *    - entry_DS = [domain_separator_label || proof.num_vars || proof.degree || public || claimed_sum]
 *
 * 2. First Round:
 *    - entry_0 = [round_poly_label || r_0[x].len() || m=0 || r_0[x]]
 *    - alpha_0 = Hash(entry_DS || seed_rng || round_challenge_label || entry_0).to_field()
 *
 * 3. Round i > 0:
 *    - entry_i = [round_poly_label || r_i[x].len() || m=i || r_i[x]]
 *    - alpha_i = Hash(entry_0 || alpha_(i-1) || round_challenge_label || entry_i).to_field()
 *
 * Notes:
 * - Both Prover and Verifier use this configuration to ensure deterministic randomness generation.
 * - Messages (e.g., field elements, strings) are encoded in little/big-endian format based on
 *   `SumcheckTranscriptConfig.m_little_endian`.
 */

namespace icicle {

  /**
   * @brief Configuration for encoding and hashing prover messages in the Sumcheck protocol.
   *
   * This class describes how prover messages are encoded and hashed to deterministically
   * generate randomness. It includes:
   * - Labels for various protocol stages (e.g., domain separator, round polynomials, challenges).
   * - Endianness for encoding messages (little-endian by default).
   * - The hashing function used to derive randomness.
   * - A seed for initializing the random number generator (RNG).
   *
   * @tparam F Type of the field element (e.g., prime field or extension field elements).
   */
  template <typename F>
  class SumcheckTranscriptConfig
  {
  public:
  public:
    // Default Constructor
    SumcheckTranscriptConfig() : m_little_endian(true), m_seed_rng(0) {}

    // Constructor with byte vector for labels
    SumcheckTranscriptConfig(
      Hash hasher,
      std::vector<std::byte>&& domain_label,
      std::vector<std::byte>&& poly_label,
      std::vector<std::byte>&& challenge_label,
      F seed,
      bool little_endian = true)
        : m_hasher(std::move(hasher)), m_domain_separator_label(domain_label), m_round_poly_label(poly_label),
          m_round_challenge_label(challenge_label), m_little_endian(little_endian), m_seed_rng(seed)
    {
    }

    // Constructor with const char* arguments for labels
    SumcheckTranscriptConfig(
      Hash hasher,
      const char* domain_label,
      const char* poly_label,
      const char* challenge_label,
      F seed,
      bool little_endian = true)
        : m_hasher(std::move(hasher)), m_domain_separator_label(cstr_to_bytes(domain_label)),
          m_round_poly_label(cstr_to_bytes(poly_label)), m_round_challenge_label(cstr_to_bytes(challenge_label)),
          m_little_endian(little_endian), m_seed_rng(seed)
    {
    }

    // Move Constructor
    SumcheckTranscriptConfig(SumcheckTranscriptConfig&& other) noexcept
        : m_domain_separator_label(std::move(other.m_domain_separator_label)),
          m_round_poly_label(std::move(other.m_round_poly_label)),
          m_round_challenge_label(std::move(other.m_round_challenge_label)), m_little_endian(other.m_little_endian),
          m_seed_rng(other.m_seed_rng)
    {
    }

    // Accessors
    const std::vector<std::byte>& get_domain_separator_label() const { return m_domain_separator_label; }
    const std::vector<std::byte>& get_round_poly_label() const { return m_round_poly_label; }
    const std::vector<std::byte>& get_round_challenge_label() const { return m_round_challenge_label; }
    bool is_little_endian() const { return m_little_endian; }
    const F& get_seed_rng() const { return m_seed_rng; }

  private:
    Hash m_hasher;                                         ///< Hash function used for randomness generation.
    const std::vector<std::byte> m_domain_separator_label; ///< Label for the domain separator in the transcript.
    const std::vector<std::byte> m_round_poly_label;       ///< Label for round polynomials in the transcript.
    const std::vector<std::byte> m_round_challenge_label;  ///< Label for round challenges in the transcript.
    const bool m_little_endian = true;                     ///< Encoding endianness (default: little-endian).
    F m_seed_rng;                                          ///< Seed for initializing the RNG.

    static inline std::vector<std::byte> cstr_to_bytes(const char* str)
    {
      if (str == nullptr) return {};
      const size_t length = std::strlen(str);
      return {reinterpret_cast<const std::byte*>(str), reinterpret_cast<const std::byte*>(str) + length};
    }
  };

} // namespace icicle