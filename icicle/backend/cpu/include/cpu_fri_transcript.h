#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cstring>
#include <algorithm>
#include "icicle/fri/fri_transcript_config.h"
#include "icicle/errors.h"
#include "icicle/hash/hash.h"

namespace icicle {

template <typename F>
class CpuFriTranscript
{
public:
  CpuFriTranscript(FriTranscriptConfig<F>&& transcript_config, const size_t log_input_size)
    : m_transcript_config(std::move(transcript_config))
    , m_log_input_size(log_input_size)
    , m_prev_alpha(F::zero())
    , m_first_round(true)
    , m_pow_nonce(0)
  {
    m_entry_0.clear();
    m_first_round = true;
  }

  /**
    * @brief Add a Merkle commit to the transcript and generate a new alpha challenge.
    *
    * @param merkle_commit The raw bytes of the Merkle commit.
    * @return A field element alpha derived via Fiat-Shamir.
    */
  F get_alpha(const std::vector<std::byte>& merkle_commit)
  {
    ICICLE_ASSERT(m_transcript_config.get_domain_separator_label().size() > 0) << "Domain separator label must be set";
    // Prepare a buffer for hashing
    m_entry_0.reserve(1024); // pre-allocate some space
    std::vector<std::byte> hash_input;
    hash_input.reserve(1024); // pre-allocate some space

    // Build the round's hash input
    if (m_first_round) {
      build_entry_0();
      build_hash_input_round_0(hash_input);
      m_first_round = false;
    } else {
      build_hash_input_round_i(hash_input);
    }
    append_data(hash_input, merkle_commit);

    // Hash the input and return alpha
    const Hash& hasher = m_transcript_config.get_hasher();
    std::vector<std::byte> hash_result(hasher.output_size());
    hasher.hash(hash_input.data(), hash_input.size(), m_hash_config, hash_result.data());
    reduce_hash_result_to_field(m_prev_alpha, hash_result);
    return m_prev_alpha;
  }

  size_t hash_and_get_nof_leading_zero_bits(uint64_t nonce, const size_t pow_bits)
  {
    // Prepare a buffer for hashing
    std::vector<std::byte> hash_input;
    hash_input.reserve(1024); // pre-allocate some space

    // Build the hash input
    build_hash_input_pow(hash_input);

    const Hash& hasher = m_transcript_config.get_hasher();
    std::vector<std::byte> hash_result(hasher.output_size());
    hasher.hash(hash_input.data(), hash_input.size(), m_hash_config, hash_result.data());

    return count_leading_zero_bits(hash_result);
  }

  /**
    * @brief Add a proof-of-work nonce to the transcript, to be included in subsequent rounds.
    * @param pow_nonce The proof-of-work nonce.
    */
  void set_pow_nonce(uint32_t pow_nonce)
  {
    m_pow_nonce = pow_nonce;
  }

  size_t get_seed_for_query_phase()
  {
    // Prepare a buffer for hashing
    std::vector<std::byte> hash_input;
    hash_input.reserve(1024); // pre-allocate some space

    // Build the hash input
    build_hash_input_query_phase(hash_input);

    const Hash& hasher = m_transcript_config.get_hasher();
    std::vector<std::byte> hash_result(hasher.output_size());
    hasher.hash(hash_input.data(), hash_input.size(), m_hash_config, hash_result.data());
    uint64_t seed = bytes_to_uint_64(hash_result);
    return seed;
  }

private:
  const FriTranscriptConfig<F> m_transcript_config; // Transcript configuration (labels, seeds, etc.)
  const size_t m_log_input_size;                    // Logarithm of the initial input size
  const HashConfig m_hash_config;                   // hash config - default
  bool m_first_round;                               // Indicates if this is the first round
  std::vector<std::byte> m_entry_0;                 // Hash input set in the first round and used in all subsequent rounds
  F m_prev_alpha;                                   // The previous alpha generated
  uint64_t m_pow_nonce;                             // Proof-of-work nonce - optional

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
    * @brief Append an unsigned 64-bit integer to the byte vector (little-endian).
    * @param dest (OUT) Destination byte vector.
    * @param value The 64-bit value to append.
    */
  void append_u32(std::vector<std::byte>& dest, uint32_t value)
  {
    const std::byte* data_bytes = reinterpret_cast<const std::byte*>(&value);
    dest.insert(dest.end(), data_bytes, data_bytes + sizeof(uint32_t));
  }

  /**
    * @brief Append a field element to the byte vector.
    * @param dest (OUT) Destination byte vector.
    * @param field The field element to append.
    */
  void append_field(std::vector<std::byte>& dest, const F& field)
  {
    const std::byte* data_bytes = reinterpret_cast<const std::byte*>(field.limbs_storage.limbs);
    dest.insert(dest.end(), data_bytes, data_bytes + sizeof(F));
  }

  /**
    * @brief Convert a hash output into a field element by copying a minimal number of bytes.
    * @param alpha (OUT) The resulting field element.
    * @param hash_result A buffer of bytes (from the hash function).
    */
  void reduce_hash_result_to_field(F& alpha, const std::vector<std::byte>& hash_result)
  {
    alpha = F::zero();
    const int nof_bytes_to_copy = std::min<int>(sizeof(alpha), static_cast<int>(hash_result.size()));
    std::memcpy(&alpha, hash_result.data(), nof_bytes_to_copy);
    alpha = alpha * F::one(); 
  }


  /**
    * @brief Build the hash input for round 0 (commit phase 0).
    *
    * DS =[domain_seperator||log_2(initial_domain_size).LE32()]
    * entry_0 =[DS||public.LE32()]
    *
    */
  void build_entry_0()
  {
    append_data(m_entry_0, m_transcript_config.get_domain_separator_label());
    append_u32(m_entry_0, m_log_input_size);
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
  void build_hash_input_round_0(std::vector<std::byte>& hash_input)
  {
    append_data(hash_input, m_entry_0);
    append_field(hash_input, m_transcript_config.get_seed_rng());
    append_data(hash_input, m_transcript_config.get_round_challenge_label());
    append_data(hash_input, m_transcript_config.get_commit_phase_label());
  }

  /**
    * @brief Build the hash input for the subsequent rounds (commit phase i).
    *
    * alpha_n = hash(entry0||alpha_n-1||round_challenge_label[u8]||commit_label[u8]|| root_n.LE32()).to_ext_field()
    * root is added outside this function
    *
    * @param hash_input (OUT) The byte vector that accumulates data to be hashed.
    */
  void build_hash_input_round_i(std::vector<std::byte>& hash_input)
  {
    append_data(hash_input, m_entry_0);
    append_field(hash_input, m_prev_alpha);
    append_data(hash_input, m_transcript_config.get_round_challenge_label());
    append_data(hash_input, m_transcript_config.get_commit_phase_label());
  }

  /**
    * @brief Build the hash input for the proof-of-work nonce.
    * hash_input = entry_0||alpha_{n-1}||"nonce"||nonce
    *
    * @param hash_input (OUT) The byte vector that accumulates data to be hashed.
    */
    void build_hash_input_pow(std::vector<std::byte>& hash_input, uint32_t temp_pow_nonce)
  {
    append_data(hash_input, m_entry_0);
    append_field(hash_input, m_prev_alpha);
    append_data(hash_input, m_transcript_config.get_nonce_label());
    append_u32(hash_input, temp_pow_nonce);
  }

  /**
    * @brief Build the hash input for the query phase.
    * hash_input = entry_0||alpha_{n-1}||"query"||seed
    *
    * @param hash_input (OUT) The byte vector that accumulates data to be hashed.
    */
  void build_hash_input_query_phase(std::vector<std::byte>& hash_input)
  {
    if (m_pow_nonce ==0){
      append_data(hash_input, m_entry_0);
      append_field(hash_input, m_prev_alpha);
    } else {
      append_data(hash_input, m_entry_0);
      append_data(hash_input, m_transcript_config.get_nonce_label());
      append_u32(hash_input, m_pow_nonce);
    }
  }

  static size_t count_leading_zero_bits(const std::vector<std::byte>& data)
  {
    size_t zero_bits = 0;
    for (size_t i = 0; i < data.size(); i++) {
      uint8_t byte_val = static_cast<uint8_t>(data[i]);
      if (byte_val == 0) {
        zero_bits += 8;
      } else {
        for (int bit = 7; bit >= 0; bit--) {
          if ((byte_val & (1 << bit)) == 0) {
            zero_bits++;
          } else {
            return zero_bits;
          }
        }
        break;
      }
    }
    return zero_bits;
  }


  uint64_t bytes_to_uint_64(const std::vector<std::byte>& data)
  {
    uint64_t result = 0;
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
      result |= static_cast<uint64_t>(data[i]) << (i * 8);
    }
    return result;
  }

};

} // namespace icicle
