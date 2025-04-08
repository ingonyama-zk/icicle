#pragma once
#include "icicle/sumcheck/sumcheck_transcript_config.h"
#include <string.h>

template <typename S>
class SumcheckTranscript
{
public:
  SumcheckTranscript(
    const S& claimed_sum,
    const uint32_t mle_polynomial_size,
    const uint32_t combine_function_poly_degree,
    SumcheckTranscriptConfig<S>&& transcript_config)
      : m_claimed_sum(claimed_sum), m_mle_polynomial_size(mle_polynomial_size),
        m_combine_function_poly_degree(combine_function_poly_degree), m_transcript_config(std::move(transcript_config))
  {
    // Check inputs
    ICICLE_ASSERT(m_mle_polynomial_size > 0) << "mle_polynomial_size must be > 0";
    ICICLE_ASSERT(m_combine_function_poly_degree > 0) << "combine_function_poly_degree must be > 0";
    m_entry_0.clear();
    m_round_idx = 0;
  }

  // add round polynomial to the transcript
  S get_alpha(const std::vector<S>& round_poly)
  {
    const std::vector<std::byte>& round_poly_label = m_transcript_config.get_round_poly_label();
    std::vector<std::byte> hash_input;
    hash_input.reserve(2048);
    (m_round_idx == 0) ? build_hash_input_round_0(hash_input, round_poly)
                       : build_hash_input_round_i(hash_input, round_poly);
    print_byte_vector("CPP: hash_input DS", hash_input);
    // hash hash_input and return alpha
    const Hash& hasher = m_transcript_config.get_hasher();
    std::vector<std::byte> hash_result(hasher.output_size());
    hasher.hash(hash_input.data(), hash_input.size(), m_config, hash_result.data());
    std::cout << "round: " << m_round_idx << std::endl;
    print_byte_vector("CPP: hash_result", hash_result);
    m_round_idx++;
    m_prev_alpha = S::from(hash_result.data(), hasher.output_size());
    return m_prev_alpha;
  }

private:
  const SumcheckTranscriptConfig<S> m_transcript_config; // configuration how to build the transcript
  HashConfig m_config;                                   // hash config - default
  uint32_t m_round_idx;                                  //
  std::vector<std::byte> m_entry_0;                      //
  uint32_t m_mle_polynomial_size = 0;
  uint32_t m_combine_function_poly_degree = 0;
  const S m_claimed_sum;
  S m_prev_alpha;

  // append to hash_input a stream of bytes received as chars
  void append_data(std::vector<std::byte>& byte_vec, const std::vector<std::byte>& label)
  {
    byte_vec.insert(byte_vec.end(), label.begin(), label.end());
  }

  // append an integer uint32_t to hash input
  void append_u32(std::vector<std::byte>& byte_vec, const uint32_t data)
  {
    const std::byte* data_bytes = reinterpret_cast<const std::byte*>(&data);
    byte_vec.insert(byte_vec.end(), data_bytes, data_bytes + sizeof(uint32_t));
  }

  // append a field to hash input
  void append_field(std::vector<std::byte>& byte_vec, const S& field)
  {
    const std::byte* data_bytes = reinterpret_cast<const std::byte*>(field.limbs_storage.limbs);
    byte_vec.insert(byte_vec.end(), data_bytes, data_bytes + sizeof(S));
  }
  
  void print_byte_vector(const std::string& label, const std::vector<std::byte>& vec) {
    std::cout << label << " (len = " << vec.size() << ") = 0x";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << static_cast<int>(vec[i]); // Convert to int for printing
    }
    std::cout << std::dec << std::endl; // Reset to decimal for any subsequent prints
}

  // round 0 hash input
  void build_hash_input_round_0(std::vector<std::byte>& hash_input, const std::vector<S>& round_poly)
  {
    const std::vector<std::byte>& round_poly_label = m_transcript_config.get_round_poly_label();
    // append entry_DS = [domain_separator_label || proof.mle_polynomial_size || proof.degree || public (hardcoded?) ||
    // claimed_sum]
    append_data(hash_input, m_transcript_config.get_domain_separator_label());
    //print_byte_vector("hash_input DS", hash_input);
    append_u32(hash_input, m_mle_polynomial_size);
    //print_byte_vector("hash_input D||size", hash_input);
    append_u32(hash_input, m_combine_function_poly_degree);
    //print_byte_vector("hash_input D||size||deg", hash_input);
    append_field(hash_input, m_claimed_sum);
    //print_byte_vector("hash_input D||size||deg|claimedsum ", hash_input);

    // append seed_rng
    append_field(hash_input, m_transcript_config.get_seed_rng());
    //print_byte_vector("hash_input D||size||deg|claimedsum||seedrng ", hash_input);

    // append round_challenge_label
    append_data(hash_input, m_transcript_config.get_round_challenge_label());
    //print_byte_vector("hash_input D||size||deg|claimedsum||seedrng||ch_label", hash_input);

    // build entry_0 = [round_poly_label || r_0[x].len() || k=0 || r_0[x]]
    append_data(m_entry_0, round_poly_label);
    append_u32(m_entry_0, round_poly.size());
    append_u32(m_entry_0, m_round_idx);
    for (const S& r_i : round_poly) {
      append_field(hash_input, r_i);
    }
    print_byte_vector("CPP: entry_0", m_entry_0);
    // append entry_0
    append_data(hash_input, m_entry_0);
    //print_byte_vector("DS||log_poly_size ||deg||seedrng||rngch_label||round_poly_labe||r0len||0||r0[0]||r0[1]", hash_input);
  }

  // round !=0 hash input
  void build_hash_input_round_i(std::vector<std::byte>& hash_input, const std::vector<S>& round_poly)
  {
    const std::vector<std::byte>& round_poly_label = m_transcript_config.get_round_poly_label();
    // entry_i = [round_poly_label || r_i[x].len() || k=i || r_i[x]]
    // alpha_i = Hash(entry_0 || alpha_(i-1) || round_challenge_label || entry_i).to_field()
    append_data(hash_input, m_entry_0);
    append_field(hash_input, m_prev_alpha);
    append_data(hash_input, m_transcript_config.get_round_challenge_label());

    append_data(hash_input, round_poly_label);
    append_u32(hash_input, round_poly.size());
    append_u32(hash_input, m_round_idx);
    for (const S& r_i : round_poly) {
      append_field(hash_input, r_i);
    }
  }
};
