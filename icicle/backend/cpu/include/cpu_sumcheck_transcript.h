



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
  


template <typename S>
class CpuSumCheckTranscript {
public:
  CpuSumCheckTranscript(const uint32_t num_vars, 
                        const uint32_t poly_degree, 
                        const S& claimed_sum, 
                        SumcheckTranscriptConfig& transcript_config) :
    m_num_vars(num_vars), 
    m_poly_degree(poly_degree),
    m_claimed_sum(claimed_sum),
    m_transcript_config(transcript_config) {
      reset();
    }

  // add round polynomial to the transcript
  S get_alpha(const vector <S>& round_poly) {
    const std::vector<std::byte>& round_poly_label = transcript_config.m_round_poly_label()
    vector<std::byte> hash_input;
    (poly_degree == 0) ? build_hash_input_round_0(hash_input, round_poly) :
                         build_hash_input_round_i(hash_input, round_poly);


    // hash hash_input and return alpha
    vector<std::byte> hash_result(transcript_config.hasher.output_size());
    m_transcript_config.hasher.hash(hash_input.data(), hash_input.size(), m_config, hash_result.data());
    m_prev_alpha = S::reduce(hash_result.data()); TODO fix that
    return m_prev_alpha;

  }    




  // reset the transcript
  voiud reset() {
    m_hash_input.clear();
    m_entry_0.clear();
    m_round_idx = 0;
  }

private:
  SumcheckTranscriptConfig& m_transcript_config;    // configuration how to build the transcript
  HashConfig                m_config;               // hash config - default
  uint32_t                  m_round_idx;            // 
  vector<std::byte>         m_entry_0;              // 
  const uint32_t            m_num_vars;
  const uint32_t            m_poly_degree;
  const S                   m_claimed_sum;
  S                         m_prev_alpha;


  // append to hash_input a stream of bytes received as chars
  void append_data(vector<std::byte>& byte_vec, const std::vector<std::byte>& label) {
    byte_vec.insert(byte_vec.end(), label.begin(), label.end());
  }

  // append an integer uint32_t to hash input
  void append_u32(vector<std::byte>& byte_vec, const uint32_t data) {
    const std::byte* data_bytes = reinterpret_cast<const std::byteuint8_t*>(&data);
    byte_vec.insert(byte_vec.end(), data_bytes, data_bytes + sizeof(uint32_t));
  }

  void append_field(vector<std::byte>& byte_vec, const S& field) {
    const std::byte* data_bytes = reinterpret_cast<const std::byteuint8_t*>(field.get_limbs());
    byte_vec.insert(byte_vec.end(), data_bytes, data_bytes + sizeof(S));
  }


  void build_hash_input_round_0(vector<std::byte>& hash_input, const vector <S>& round_poly) {
    const std::vector<std::byte>& round_poly_label = transcript_config.m_round_poly_label()
    // append entry_DS = [domain_separator_label || proof.num_vars || proof.degree || public (hardcoded?) || claimed_sum]
    append_data(hash_input, m_transcript_config.get_domain_separator_label());
    append_u32(hash_input, m_num_vars);
    append_u32(hash_input, m_poly_degree);
    append_field(hash_input, m_claimed_sum);

    // append seed_rng
    append_data(hash_input, m_transcript_config.get_seed_rng());

    // append round_challenge_label
    append_data(hash_input, m_transcript_config.get_round_challenge_label());

    // build entry_0 = [round_poly_label || r_0[x].len() || k=0 || r_0[x]]
    append_data(m_entry_0, round_poly_label);
    append_u32(m_entry_0, round_poly.size());
    append_u32(m_entry_0, m_round_idx++);
    for (S& r_i :round_poly) {
      append_field(r_i, S);
    }

    // append entry_0 
    append_data(hash_input, m_entry_0);
  }
  void build_hash_input_round_i(vector<std::byte>& hash_input, const vector <S>& round_poly) {
    // entry_i = [round_poly_label || r_i[x].len() || k=i || r_i[x]]
    // alpha_i = Hash(entry_0 || alpha_(i-1) || round_challenge_label || entry_i).to_field()
    append_data(hash_input, m_entry_0);
    append_field(hash_input, m_prev_alpha);
    append_data(hash_input, m_transcript_config.get_round_challenge_label());

    append_data(hash_input, round_poly_label);
    append_u32(hash_input, round_poly.size());
    append_u32(hash_input, m_round_idx++);
    for (S& r_i :round_poly) {
      append_field(r_i, S);
    }
  }
};
