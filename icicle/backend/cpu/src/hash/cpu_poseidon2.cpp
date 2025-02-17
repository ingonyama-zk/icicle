#include "icicle/backend/hash/poseidon2_backend.h"
#include "icicle/utils/utils.h"

/// These are pre-calculated constants for different curves
#include "icicle/fields/id.h"
#if FIELD_ID == BN254
  #include "icicle/hash/poseidon2_constants/constants/bn254_poseidon2.h"
using namespace poseidon2_constants_bn254;
#elif FIELD_ID == BLS12_381
  #include "icicle/hash/poseidon2_constants/constants/bls12_381_poseidon2.h"
using namespace poseidon2_constants_bls12_381;
#elif FIELD_ID == BLS12_377
  #include "icicle/hash/poseidon2_constants/constants/bls12_377_poseidon2.h"
using namespace poseidon2_constants_bls12_377;
#elif FIELD_ID == BW6_761
  #include "icicle/hash/poseidon2_constants/constants/bw6_761_poseidon2.h"
using namespace poseidon2_constants_bw6_761;
#elif FIELD_ID == GRUMPKIN
  #include "icicle/hash/poseidon2_constants/constants/grumpkin_poseidon2.h"
using namespace poseidon2_constants_grumpkin;
#elif FIELD_ID == M31
  #include "icicle/hash/poseidon2_constants/constants/m31_poseidon2.h"
using namespace poseidon2_constants_m31;
#elif FIELD_ID == BABY_BEAR
  #include "icicle/hash/poseidon2_constants/constants/babybear_poseidon2.h"
using namespace poseidon2_constants_babybear;
#elif FIELD_ID == STARK_252
  #include "icicle/hash/poseidon2_constants/constants/stark252_poseidon2.h"
using namespace poseidon2_constants_stark252;
#elif FIELD_ID == KOALA_BEAR
  #include "icicle/hash/poseidon2_constants/constants/koalabear_poseidon2.h"
using namespace poseidon2_constants_koalabear;
#endif

namespace icicle {
  template <typename S>
  class Poseidon2BackendCPU : public HashBackend
  {
  public:
    Poseidon2BackendCPU(unsigned t, const S* domain_tag)
        : HashBackend("Poseidon2-CPU", sizeof(S), sizeof(S) * (nullptr != domain_tag ? t - 1 : t)),
          m_domain_tag_value(nullptr != domain_tag ? *domain_tag : S::zero()), m_use_domain_tag(nullptr != domain_tag),
          t(t)
    {
      init_poseidon2_constants(
        t, false /* dont use_all_zeroes_padding */, default_hash_config(), &m_poseidon2_constants);
    }

    eIcicleError init_poseidon2_constants(
      int t, bool use_all_zeroes_padding, const HashConfig& config, Poseidon2ConstantsOptions<S>* poseidon2_constants)
    {
      unsigned int alpha;
      unsigned int partial_rounds;
      unsigned int full_rounds;
      unsigned int upper_full_rounds;
      unsigned int bottom_full_rounds;
      const std::string* rounds_constants;
      const std::string* mds_matrix;
      const std::string* partial_matrix_diagonal;
      unsigned int T = t;
      switch (t) {
      case 2:
        alpha = alpha_2;
        rounds_constants = rounds_constants_2;
        mds_matrix = mds_matrix_2;
        partial_matrix_diagonal = partial_matrix_diagonal_2;
        partial_rounds = partial_rounds_2;
        full_rounds = full_rounds_2;
        upper_full_rounds = half_full_rounds_2;
        bottom_full_rounds = half_full_rounds_2;
        break;
      case 3:
        alpha = alpha_3;
        rounds_constants = rounds_constants_3;
        mds_matrix = mds_matrix_3;
        partial_matrix_diagonal = partial_matrix_diagonal_3;
        partial_rounds = partial_rounds_3;
        full_rounds = full_rounds_3;
        upper_full_rounds = half_full_rounds_3;
        bottom_full_rounds = half_full_rounds_3;
        break;
      case 4:
        alpha = alpha_4;
        rounds_constants = rounds_constants_4;
        mds_matrix = mds_matrix_4;
        partial_matrix_diagonal = partial_matrix_diagonal_4;
        partial_rounds = partial_rounds_4;
        full_rounds = full_rounds_4;
        upper_full_rounds = half_full_rounds_4;
        bottom_full_rounds = half_full_rounds_4;
        break;
      case 8:
        alpha = alpha_8;
        rounds_constants = rounds_constants_8;
        mds_matrix = mds_matrix_8;
        partial_matrix_diagonal = partial_matrix_diagonal_8;
        partial_rounds = partial_rounds_8;
        full_rounds = full_rounds_8;
        upper_full_rounds = half_full_rounds_8;
        bottom_full_rounds = half_full_rounds_8;
        break;
      case 12:
        alpha = alpha_12;
        rounds_constants = rounds_constants_12;
        mds_matrix = mds_matrix_12;
        partial_matrix_diagonal = partial_matrix_diagonal_12;
        partial_rounds = partial_rounds_12;
        full_rounds = full_rounds_12;
        upper_full_rounds = half_full_rounds_12;
        bottom_full_rounds = half_full_rounds_12;
        break;
      case 16:
        alpha = alpha_16;
        rounds_constants = rounds_constants_16;
        mds_matrix = mds_matrix_16;
        partial_matrix_diagonal = partial_matrix_diagonal_16;
        partial_rounds = partial_rounds_16;
        full_rounds = full_rounds_16;
        upper_full_rounds = half_full_rounds_16;
        bottom_full_rounds = half_full_rounds_16;
        break;
      case 20:
        alpha = alpha_20;
        rounds_constants = rounds_constants_20;
        mds_matrix = mds_matrix_20;
        partial_matrix_diagonal = partial_matrix_diagonal_20;
        partial_rounds = partial_rounds_20;
        full_rounds = full_rounds_20;
        upper_full_rounds = half_full_rounds_20;
        bottom_full_rounds = half_full_rounds_20;
        break;
      case 24:
        alpha = alpha_24;
        rounds_constants = rounds_constants_24;
        mds_matrix = mds_matrix_24;
        partial_matrix_diagonal = partial_matrix_diagonal_24;
        partial_rounds = partial_rounds_24;
        full_rounds = full_rounds_24;
        upper_full_rounds = half_full_rounds_24;
        bottom_full_rounds = half_full_rounds_24;
        break;
      default:
        ICICLE_LOG_ERROR
          << "cpu_poseidon2_init_default_constants: t (width) must be one of [2, 3, 4, 8, 12, 16, 20, 24]";
        return eIcicleError::INVALID_ARGUMENT;
      } // switch (t) {
      if (full_rounds == 0 && partial_rounds == 0) { // All arrays are empty in this case (true for wide fields (width >
                                                     // 32) & t > 8).
        return eIcicleError::SUCCESS;
      }

      scalar_t* scalar_rounds_constants = new scalar_t[full_rounds * t + partial_rounds];
      for (int i = 0; i < (full_rounds * t + partial_rounds); i++) {
        scalar_rounds_constants[i] = scalar_t::hex_str2scalar(rounds_constants[i]);
      }
      scalar_t* scalar_mds_matrix = new scalar_t[t * t];
      for (int i = 0; i < (t * t); i++) {
        scalar_mds_matrix[i] = scalar_t::hex_str2scalar(mds_matrix[i]);
      }
      scalar_t* scalar_partial_matrix_diagonal = new scalar_t[t];
      scalar_t* scalar_partial_matrix_diagonal_m1 = new scalar_t[t];
      for (int i = 0; i < t; i++) {
        scalar_partial_matrix_diagonal[i] = scalar_t::hex_str2scalar(partial_matrix_diagonal[i]);
        scalar_partial_matrix_diagonal_m1[i] = scalar_partial_matrix_diagonal[i] - scalar_t::from(1);
      }

      poseidon2_constants->t = t;
      poseidon2_constants->alpha = alpha;
      poseidon2_constants->use_all_zeroes_padding = use_all_zeroes_padding;
      poseidon2_constants->nof_upper_full_rounds = upper_full_rounds;
      poseidon2_constants->nof_partial_rounds = partial_rounds;
      poseidon2_constants->nof_bottom_full_rounds = bottom_full_rounds;
      poseidon2_constants->rounds_constants = scalar_rounds_constants;
      poseidon2_constants->mds_matrix = scalar_mds_matrix;
      poseidon2_constants->partial_matrix_diagonal_m1 = scalar_partial_matrix_diagonal_m1;

      return eIcicleError::SUCCESS;
    } // eIcicleError init_poseidon2_constants(

    // size - number of bytes in a single inputs.
    // The total size of the input should be size * config.batch.
    eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const override
    {
      bool is_unsupported_T_for_this_field = m_poseidon2_constants.nof_upper_full_rounds == 0;
      if (is_unsupported_T_for_this_field) {
        ICICLE_LOG_ERROR << "Unsupported poseidon2 width (t=" << t << ") for this field!";
        return eIcicleError::API_NOT_IMPLEMENTED;
      }

      int input_size_in_scalars = size / sizeof(S);
      bool is_sponge = input_size_in_scalars != (m_use_domain_tag ? t - 1 : t);

      // Generate padding indications.
      int sponge_nof_hashers = 0;
      int padding_size_in_scalars = 0;
      bool is_padding_needed = false;
      if (is_sponge) {
        if (input_size_in_scalars < t) { // Single hasher in the chain.
          sponge_nof_hashers = 1;
          is_padding_needed = true;
          padding_size_in_scalars = t - (input_size_in_scalars + (m_use_domain_tag == true));
        } else { // More than a single hasher in the chain.
          sponge_nof_hashers = (input_size_in_scalars - !(m_use_domain_tag == true) + (t - 2)) / (t - 1);
          is_padding_needed = (input_size_in_scalars - !(m_use_domain_tag == true)) % (t - 1);
          if (is_padding_needed) {
            padding_size_in_scalars = (t - 1) - ((input_size_in_scalars - !(m_use_domain_tag == true)) % (t - 1));
          }
        }
      } // if (is_sponge) {

#define PERMUTATION_SPONGE_T(T)                                                                                        \
  case T:                                                                                                              \
    if constexpr (!is_large_field || T <= 8) {                                                                         \
      permutation_error = poseidon2_sponge_permutation<T>(                                                             \
        in_fields, input_size_in_scalars, is_padding_needed, padding_size_in_scalars, m_use_domain_tag,                \
        m_domain_tag_value, out, sponge_nof_hashers, config.batch, m_poseidon2_constants);                             \
    }                                                                                                                  \
    break;

#define PERMUTATION_T(T)                                                                                               \
  case T:                                                                                                              \
    if constexpr (!is_large_field || T <= 8) {                                                                         \
      permutation_error = poseidon2_permutation<T>(                                                                    \
        in_fields, m_use_domain_tag, m_domain_tag_value, out, config.batch, m_poseidon2_constants);                    \
    }                                                                                                                  \
    break;

      const S* in_fields = (S*)(input);
      S* out = (S*)(output);
      constexpr bool is_large_field = sizeof(S) > 8;
      eIcicleError permutation_error = eIcicleError::SUCCESS;

      if (is_sponge) {
        switch (t) {
          PERMUTATION_SPONGE_T(2);
          PERMUTATION_SPONGE_T(3);
          PERMUTATION_SPONGE_T(4);
          PERMUTATION_SPONGE_T(8);
          PERMUTATION_SPONGE_T(12);
          PERMUTATION_SPONGE_T(16);
          PERMUTATION_SPONGE_T(20);
          PERMUTATION_SPONGE_T(24);
        }
      } else {
        switch (t) {
          PERMUTATION_T(2);
          PERMUTATION_T(3);
          PERMUTATION_T(4);
          PERMUTATION_T(8);
          PERMUTATION_T(12);
          PERMUTATION_T(16);
          PERMUTATION_T(20);
          PERMUTATION_T(24);
        }
      }

      return permutation_error;
    } // eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const
      // override

  private:
    template <int T>
    void prepare_poseidon2_states(const S* input, S* states, bool use_domain_tag, S domain_tag) const
    {
      S prepared_element;
#pragma unroll
      for (int element_idx_in_hash = 0; element_idx_in_hash < T; element_idx_in_hash++) {
        if (use_domain_tag) {
          if (element_idx_in_hash == 0) {
            prepared_element = domain_tag;
          } else {
            prepared_element = input[element_idx_in_hash - 1];
          }
        } else {
          prepared_element = input[element_idx_in_hash];
        }
        states[element_idx_in_hash] = prepared_element;
      }
    } // prepare_poseidon2_states(const S* input, S* states, bool use_domain_tag, S domain_tag)

    template <int T>
    void prepare_poseidon2_sponge_states(
      const S*& input,
      const bool is_padding_needed,
      const S* padding,
      const int padding_size_in_scalars,
      S* states,
      const bool is_last_hasher) const
    {
#pragma unroll
      for (int states_idx = 1; states_idx < T; states_idx++) {
        if (is_last_hasher && is_padding_needed) {
          int nof_valid_inputs_in_hashers = (T - 1) - padding_size_in_scalars;
          if (states_idx < 1 + nof_valid_inputs_in_hashers)
            states[states_idx] = states[states_idx] + input[states_idx - 1];
          else
            states[states_idx] = states[states_idx] + padding[states_idx - (1 + nof_valid_inputs_in_hashers)];
        } else { // Not last hasher or no padding.
          states[states_idx] = states[states_idx] + input[states_idx - 1];
        }
      }
    } // prepare_poseidon2_sponge_states(

    // This function performs a full matrix by vector multiplication.
    template <int T>
    void full_matrix_mul_vec(const S* vec_in, const S* matrix_in, S* result) const
    {
      S tmp_col_res[T]; // Have to use temp storage because vec_in and result are the same storage.
#pragma unroll
      for (int row_idx = 0; row_idx < T; row_idx++) { // Rows of matrix.
        tmp_col_res[row_idx] = matrix_in[row_idx * T] * vec_in[0];
#pragma unroll
        for (int col_idx = 1; col_idx < T; col_idx++) { // Columns of matrix.
          tmp_col_res[row_idx] = tmp_col_res[row_idx] + matrix_in[row_idx * T + col_idx] * vec_in[col_idx];
        }
      }
#pragma unroll
      for (int col_idx = 0; col_idx < T; col_idx++) { // This copy is needed because vec_in and result storages are
                                                      // actually the same storage when calling to the function.
        result[col_idx] = tmp_col_res[col_idx];
      }
    } // eIcicleError hash_single(const std::byte* input, std::byte* output) const

    template <int T>
    void pre_full_round(S* states, const Poseidon2ConstantsOptions<S> constants) const
    {
      // Multiply mds matrix by the states vector.
      S* matrix = constants.mds_matrix;
      full_matrix_mul_vec<T>(states, matrix, states);
    }

    template <int T>
    void full_round(S* states, size_t rc_offset, const S* rounds_constants, const S* mds_matrix, const int alpha) const
    {
#pragma unroll
      for (int element_idx_in_hash = 0; element_idx_in_hash < T; element_idx_in_hash++) {
        states[element_idx_in_hash] = states[element_idx_in_hash] + rounds_constants[rc_offset + element_idx_in_hash];
      }
#pragma unroll
      for (int element_idx_in_hash = 0; element_idx_in_hash < T; element_idx_in_hash++) {
        states[element_idx_in_hash] = S::pow(states[element_idx_in_hash], alpha);
      }

      full_matrix_mul_vec<T>(states, mds_matrix, states);
    }

    template <int T>
    void full_rounds(
      S* states,
      size_t rc_offset,
      const int nof_upper_full_rounds,
      const S* rounds_constants,
      const S* mds_matrix,
      const int alpha) const
    {
      for (int i = 0; i < nof_upper_full_rounds; i++) {
        full_round<T>(states, rc_offset, rounds_constants, mds_matrix, alpha);
        rc_offset += T;
      }
    }

    template <int T>
    void partial_round(
      S state[T], size_t rc_offset, const S* rounds_constants, const S* partial_matrix_diagonal_m1, const int alpha)
      const
    {
      state[0] = state[0] + rounds_constants[rc_offset];

      state[0] = S::pow(state[0], alpha);

      // Multiply partial matrix by vector.
      // Partial matrix is represented by T members - diagonal members of the matrix.
      // The values are actual values minus 1 in order to gain performance.
      S vec_in_sum = state[0];
#pragma unroll
      for (int i = 1; i < T; i++) {
        vec_in_sum = vec_in_sum + state[i];
      }
#pragma unroll
      for (int i = 0; i < T; i++) {
        state[i] = vec_in_sum + (partial_matrix_diagonal_m1[i] * state[i]);
      }
      for (int i = 0; i < T; i++) {}
    } // void partial_round(S state[T], size_t rc_offset, const Poseidon2ConstantsOptions<S>& constants)

    template <int T>
    // void partial_rounds(S* states, size_t rc_offset, const Poseidon2ConstantsOptions<S> constants) const
    void partial_rounds(
      S* states,
      size_t rc_offset,
      const S* rounds_constants,
      const S* partial_matrix_diagonal_m1,
      const int nof_partial_rounds,
      const int alpha) const
    {
      for (int i = 0; i < nof_partial_rounds; i++) {
        partial_round<T>(states, rc_offset, rounds_constants, partial_matrix_diagonal_m1, alpha);
        rc_offset++;
      }
    }

    template <int T>
    void squeeze_states(const S* states, unsigned int offset, S* out) const
    {
      out[0] = states[offset];
    }

    template <int T>
    eIcicleError poseidon2_permutation(
      const S* input,
      bool use_domain_tag,
      const S domain_tag_value,
      S* out,
      unsigned int batch_size,
      const Poseidon2ConstantsOptions<S>& constants) const
    {
      S* states = new S[T];

      for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        prepare_poseidon2_states<T>(input, states, use_domain_tag, domain_tag_value);

        pre_full_round<T>(states, constants);

        size_t rc_offset = 0;

        full_rounds<T>(
          states, rc_offset, constants.nof_upper_full_rounds, constants.rounds_constants, constants.mds_matrix,
          constants.alpha);
        rc_offset += T * constants.nof_upper_full_rounds;

        partial_rounds<T>(
          states, rc_offset, constants.rounds_constants, constants.partial_matrix_diagonal_m1,
          constants.nof_partial_rounds, constants.alpha);
        rc_offset += constants.nof_partial_rounds;

        full_rounds<T>(
          states, rc_offset, constants.nof_upper_full_rounds, constants.rounds_constants, constants.mds_matrix,
          constants.alpha);

        out[0] = states[1];

        input += use_domain_tag ? T - 1 : T; // Move to the next hasher input.
        out += 1;                            // Move to the output of the next hasher.
      } // for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {

      return eIcicleError::SUCCESS;
    } // eIcicleError poseidon2_permutation()

    template <int T>
    eIcicleError poseidon2_sponge_permutation(
      const S* input,
      const int input_size_in_scalars,
      const bool is_padding_needed,
      const int padding_size_in_scalars,
      const bool use_domain_tag,
      const S domain_tag_value,
      S* out,
      int nof_hashers,
      unsigned int batch_size,
      const Poseidon2ConstantsOptions<S>& constants) const
    {
      S padding[T];
      padding[0] = S::from(1);
      for (int i = 1; i < T; i++)
        padding[i] = S::from(0);

      for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        const S* input_shadow = input;

        // states vector should be set to zero in order to use addition in prepare_poseidon2_sponge_states.
        S states[T];
        for (int i = 0; i < T; i++)
          states[i] = S::from(0);
        // Take care of first input of the first hasher. Rest of the input has T-1 granularity.
        if (use_domain_tag) {
          states[0] = domain_tag_value;
        } else {
          states[0] = input[0];
          input += 1;
        }

        for (int hasher_idx = 0; hasher_idx < nof_hashers; hasher_idx++) {
          bool is_last_hasher = hasher_idx == nof_hashers - 1;

          prepare_poseidon2_sponge_states<T>(
            input, is_padding_needed, padding, padding_size_in_scalars, states, is_last_hasher);
          input += T - 1; // Move to the next hasher.

          pre_full_round<T>(states, constants);

          size_t rc_offset = 0;

          full_rounds<T>(
            states, rc_offset, constants.nof_upper_full_rounds, constants.rounds_constants, constants.mds_matrix,
            constants.alpha);
          rc_offset += T * constants.nof_upper_full_rounds;

          partial_rounds<T>(
            states, rc_offset, constants.rounds_constants, constants.partial_matrix_diagonal_m1,
            constants.nof_partial_rounds, constants.alpha);
          rc_offset += constants.nof_partial_rounds;

          full_rounds<T>(
            states, rc_offset, constants.nof_upper_full_rounds, constants.rounds_constants, constants.mds_matrix,
            constants.alpha);

        } // for (int hasher_idx = 0; hasher_idx < nof_hashers; hasher_idx ++) {

        out[batch_idx] = states[1];
        input = input_shadow + input_size_in_scalars; // Move to the input of the next hasher.
      } // for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {

      return eIcicleError::SUCCESS;
    } // eIcicleError poseidon2_sponge_permutation(

    const unsigned t;
    Poseidon2ConstantsOptions<S> m_poseidon2_constants;
    const bool m_use_domain_tag;
    const S m_domain_tag_value;

  }; // class Poseidon2BackendCUDA : public HashBackend

  static eIcicleError create_cpu_poseidon2_hash_backend(
    const Device& device, unsigned t, const scalar_t* domain_tag, std::shared_ptr<HashBackend>& backend /*OUT*/)
  {
    backend = std::make_shared<Poseidon2BackendCPU<scalar_t>>(t, domain_tag);
    return eIcicleError::SUCCESS;
  }

  REGISTER_CREATE_POSEIDON2_BACKEND("CPU", create_cpu_poseidon2_hash_backend);

} // namespace icicle
