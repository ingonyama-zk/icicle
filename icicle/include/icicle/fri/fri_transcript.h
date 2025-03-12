#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cstring>
#include <algorithm>
#include "icicle/fri/fri_transcript_config.h"
#include "icicle/errors.h"
#include "icicle/hash/hash.h"
#include "icicle/hash/pow.h"

#define PRE_ALLOCATED_SPACE 1024

namespace icicle {

  template <typename F>
  class FriTranscript
  {
  public:
    FriTranscript(const FriTranscriptConfig<F>& transcript_config, const uint32_t log_input_size)
        : m_transcript_config(transcript_config), m_prev_alpha(F::zero()), m_pow_nonce(0)
    {
      m_entry_0.clear();
      m_entry_0.reserve(PRE_ALLOCATED_SPACE); // pre-allocate some space
      build_entry_0(log_input_size);
    }

    /**
     * @brief Add a Merkle commit to the transcript and generate a new alpha challenge.
     *
     * @param merkle_commit The raw bytes of the Merkle commit.
     * @return A field element alpha derived via Fiat-Shamir.
     */
    F get_alpha(const std::vector<std::byte>& merkle_commit, bool is_first_round, eIcicleError& err)
    {
      std::vector<std::byte> hash_input;
      hash_input.reserve(PRE_ALLOCATED_SPACE); // pre-allocate some space

      // Build the round's hash input
      if (is_first_round) {
        build_hash_input_round_0(hash_input, merkle_commit);
      } else {
        build_hash_input_round_i(hash_input, merkle_commit);
      }

      // Hash the input and return alpha
      const Hash& hasher = m_transcript_config.get_hasher();
      std::vector<std::byte> hash_result(hasher.output_size());
      const HashConfig hash_config;
      err = hasher.hash(hash_input.data(), hash_input.size(), hash_config, hash_result.data());
      m_prev_alpha = F::from(hash_result.data(), hasher.output_size());
      return m_prev_alpha;
    }

    bool verify_pow(uint64_t nonce, uint8_t pow_bits)
    {
      // Prepare a buffer for hashing
      std::vector<std::byte> hash_input;
      hash_input.reserve(PRE_ALLOCATED_SPACE); // pre-allocate some space
      build_hash_input_pow(hash_input);
      const Hash& hasher = m_transcript_config.get_hasher();
      const PowConfig cfg;
      uint64_t mined_hash;
      bool is_correct;
      proof_of_work_verify(hasher, hash_input.data(), hash_input.size(), pow_bits, cfg, nonce, is_correct, mined_hash);

      return is_correct;
    }

    eIcicleError solve_pow(uint64_t& nonce, size_t pow_bits, bool& found)
    {
      // Prepare a buffer for hashing
      std::vector<std::byte> hash_input;
      hash_input.reserve(PRE_ALLOCATED_SPACE); // pre-allocate some space
      build_hash_input_pow(hash_input);
      const Hash& hasher = m_transcript_config.get_hasher();
      const PowConfig cfg;
      uint64_t mined_hash;
      eIcicleError pow_error =
        proof_of_work(hasher, hash_input.data(), hash_input.size(), pow_bits, cfg, found, nonce, mined_hash);

      return pow_error;
    }

    /**
     * @brief Add a proof-of-work nonce to the transcript, to be included in subsequent rounds.
     * @param pow_nonce The proof-of-work nonce.
     */
    void set_pow_nonce(uint32_t pow_nonce) { m_pow_nonce = pow_nonce; }

    /**
     * @brief Generates random query indices for the query phase.
     *        The seed is derived from the current transcript state.
     * @param nof_queries Number of query indices to generate.
     * @param min Lower limit.
     * @param max Upper limit.
     * @return Random (uniform distribution) unsigned integer s.t. min <= integer <= max.
     */
    std::vector<size_t> rand_queries_indicies(size_t nof_queries, size_t min, size_t max, eIcicleError& err)
    {
      // Prepare a buffer for hashing
      std::vector<std::byte> hash_input;
      hash_input.reserve(PRE_ALLOCATED_SPACE); // pre-allocate some space

      // Build the hash input
      build_hash_input_query_phase(hash_input);

      const Hash& hasher = m_transcript_config.get_hasher();
      std::vector<std::byte> hash_result(hasher.output_size());
      const HashConfig hash_config;
      err = hasher.hash(hash_input.data(), hash_input.size(), hash_config, hash_result.data());
      uint64_t seed = bytes_to_uint_64(hash_result);
      seed_rand_generator(seed);
      std::vector<size_t> vec(nof_queries);
      for (size_t i = 0; i < nof_queries; i++) {
        vec[i] = rand_size_t(min, max);
      }
      return vec;
    }

  private:
    const FriTranscriptConfig<F>& m_transcript_config;
    std::vector<std::byte> m_entry_0; // Hash input set in the first round and used in all subsequent rounds
    F m_prev_alpha;
    uint64_t m_pow_nonce; // Proof-of-work nonce - optional

    /**
     * @brief Append a vector of bytes to another vector of bytes.
     * @param dest (OUT) Destination byte vector.
     * @param src  Source byte vector.
     */
    void append_data(std::vector<std::byte>& dest, const std::vector<std::byte>& src)
    {
      dest.insert(dest.end(), src.begin(), src.end());
    }

    /**
     * @brief Append an integral value to the byte vector (little-endian).
     * @tparam T Type of the value.
     * @param dest (OUT) Destination byte vector.
     * @param value The 64-bit value to append.
     */
    template <typename T>
    void append_value(std::vector<std::byte>& dest, T value)
    {
      const std::byte* data_bytes = reinterpret_cast<const std::byte*>(&value);
      dest.insert(dest.end(), data_bytes, data_bytes + sizeof(T));
    }

    /**
     * @brief Append a field element to the byte vector.
     * @param dest (OUT) Destination byte vector.
     * @param field The field element to append.
     */
    void append_field(std::vector<std::byte>& dest, const F& field)
    {
      const std::byte* data_bytes = reinterpret_cast<const std::byte*>(&field);
      dest.insert(dest.end(), data_bytes, data_bytes + sizeof(F));
    }

    /**
     * @brief Build the hash input for round 0 (commit phase 0).
     *
     * DS =[domain_seperator||log_2(initial_domain_size).LE32()]
     * entry_0 =[DS||public.LE32()]
     *
     */
    void build_entry_0(uint32_t log_input_size)
    {
      append_data(m_entry_0, m_transcript_config.get_domain_separator_label());
      append_value<uint32_t>(m_entry_0, log_input_size);
      append_data(m_entry_0, m_transcript_config.get_public_state());
    }

    /**
     * @brief Build the hash input for round 0 (commit phase 0).
     *
     * alpha_0 = hash(entry_0||rng||round_challenge_label[u8]||commit_label[u8]|| root_0.LE32()).to_ext_field()
     * root is added outside this function
     *
     * @param hash_input (OUT) The byte vector that accumulates data to be hashed.
     */
    void build_hash_input_round_0(std::vector<std::byte>& hash_input, const std::vector<std::byte>& merkle_commit)
    {
      append_data(hash_input, m_entry_0);
      append_field(hash_input, m_transcript_config.get_seed_rng());
      append_data(hash_input, m_transcript_config.get_round_challenge_label());
      append_data(hash_input, m_transcript_config.get_commit_phase_label());
      append_data(hash_input, merkle_commit);
    }

    /**
     * @brief Build the hash input for the subsequent rounds (commit phase i).
     *
     * alpha_n = hash(entry0||alpha_n-1||round_challenge_label[u8]||commit_label[u8]|| root_n.LE32()).to_ext_field()
     * root is added outside this function
     *
     * @param hash_input (OUT) The byte vector that accumulates data to be hashed.
     */
    void build_hash_input_round_i(std::vector<std::byte>& hash_input, const std::vector<std::byte>& merkle_commit)
    {
      append_data(hash_input, m_entry_0);
      append_field(hash_input, m_prev_alpha);
      append_data(hash_input, m_transcript_config.get_round_challenge_label());
      append_data(hash_input, m_transcript_config.get_commit_phase_label());
      append_data(hash_input, merkle_commit);
    }

    /**
     * @brief Build the hash input prefix. The nonce is added later
     * hash_input_prefix = entry_0||alpha_{n-1}||"nonce"
     * hash_input = hash_input_prefix||nonce
     *
     * @param hash_input (OUT) The byte vector that accumulates data to be hashed.
     */
    void build_hash_input_pow(std::vector<std::byte>& hash_input)
    {
      append_data(hash_input, m_entry_0);
      append_field(hash_input, m_prev_alpha);
      append_data(hash_input, m_transcript_config.get_nonce_label());
    }

    /**
     * @brief Build the hash input for the query phase.
     *
     * - If PoW is **enabled**: `hash_input = entry_0 || "nonce" || nonce`
     * - If PoW is **disabled**: `hash_input = entry_0 || alpha_{n-1}`
     *
     * @param hash_input (OUT) The byte vector that accumulates data to be hashed.
     */
    inline void build_hash_input_query_phase(std::vector<std::byte>& hash_input)
    {
      if (m_pow_nonce == 0) {
        append_data(hash_input, m_entry_0);
        append_field(hash_input, m_prev_alpha);
      } else {
        append_data(hash_input, m_entry_0);
        append_data(hash_input, m_transcript_config.get_nonce_label());
        append_value<uint32_t>(hash_input, m_pow_nonce);
      }
    }

    uint64_t bytes_to_uint_64(const std::vector<std::byte>& data)
    {
      uint64_t result = 0;
      if (data.size() < sizeof(uint64_t)) {
        ICICLE_LOG_ERROR << "Insufficient data size for conversion to uint64_t";
      }
      std::memcpy(&result, data.data(), sizeof(uint64_t));
      return result;
    }
  };

} // namespace icicle
