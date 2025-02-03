#include "icicle/backend/hash/poseidon2_backend.h"
#include "icicle/utils/utils.h"
#include "icicle/fields/field.h"
#include <vector>

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

#define POSEIDON2_MAX_t 24

  static unsigned int poseidon2_legal_width[] = {2, 3, 4, 8, 12, 16, 20, 24};

  static Poseidon2ConstantsOptions<scalar_t>
    poseidon2_constants[POSEIDON2_MAX_t + 1]; // The size of this array is POSEIDON2_MAX_t + 1 because Poseidon2 max t
                                              // is 24. Only terms in poseidon2_legal_width are filled with data. Rest
                                              // of the terms are not relevant.
  static bool s_cpu_backend_poseidon2_constants_initialized = false;

  static eIcicleError init_default_constants()
  {
    if (s_cpu_backend_poseidon2_constants_initialized) { return eIcicleError::SUCCESS; }

    unsigned int alpha;
    unsigned int partial_rounds;
    unsigned int full_rounds;
    unsigned int upper_full_rounds;
    unsigned int bottom_full_rounds;
    const std::string* rounds_constants;
    const std::string* mds_matrix;
    const std::string* partial_matrix_diagonal;
    const std::string* partial_matrix_diagonal_m1;
    // At this stage it's still unknown what t and use_domain_tag will be used.
    // That's the reason that all the relevant members of the poseidon2_constants array are
    // loaded at this stage.
    for (int t_idx = 0; t_idx < std::size(poseidon2_legal_width); t_idx++) {
      unsigned int T = poseidon2_legal_width[t_idx]; // Single poseidon2 hash width
      switch (T) {
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
          << "cpu_poseidon2_init_default_constants: T (width) must be one of [2, 3, 4, 8, 12, 16, 20, 24]\n";
        return eIcicleError::INVALID_ARGUMENT;
      }
      if (full_rounds == 0 && partial_rounds == 0) { // All arrays are empty in this case.
        continue;
      }

      scalar_t* h_rounds_constants = new scalar_t[full_rounds * T + partial_rounds];
      for (int i = 0; i < (full_rounds * T + partial_rounds); i++) {
        h_rounds_constants[i] = scalar_t::hex_str2scalar(rounds_constants[i]);
      }

      scalar_t* h_mds_matrix = new scalar_t[T * T];
      for (int i = 0; i < (T * T); i++) {
        h_mds_matrix[i] = scalar_t::hex_str2scalar(mds_matrix[i]);
      }

      scalar_t* h_partial_matrix_diagonal = new scalar_t[T];
      scalar_t* h_partial_matrix_diagonal_m1 = new scalar_t[T];
      for (int i = 0; i < T; i++) {
        h_partial_matrix_diagonal[i] = scalar_t::hex_str2scalar(partial_matrix_diagonal[i]);
        h_partial_matrix_diagonal_m1[i] = h_partial_matrix_diagonal[i] - scalar_t::from(1);
      }

      poseidon2_constants[T].t = T;
      poseidon2_constants[T].alpha = alpha;
      poseidon2_constants[T].nof_upper_full_rounds = upper_full_rounds;
      poseidon2_constants[T].nof_bottom_full_rounds = bottom_full_rounds;
      poseidon2_constants[T].nof_partial_rounds = partial_rounds;
      poseidon2_constants[T].rounds_constants = h_rounds_constants;
      poseidon2_constants[T].mds_matrix = h_mds_matrix;
      poseidon2_constants[T].partial_matrix_diagonal_m1 = h_partial_matrix_diagonal_m1;
    } // for (int t_idx = 0; t_idx < std::size(poseidon2_legal_width); t_idx++)

    s_cpu_backend_poseidon2_constants_initialized = true;
    return eIcicleError::SUCCESS;
  }

  template <typename S>
  class Poseidon2BackendCPU : public HashBackend
  {
  public:
    Poseidon2BackendCPU(unsigned t, const S* domain_tag)
        : HashBackend("Poseidon2-CPU", sizeof(S), sizeof(S) * (nullptr != domain_tag ? t - 1 : t)),
          m_domain_tag(nullptr != domain_tag ? *domain_tag : S::zero()), m_use_domain_tag(nullptr != domain_tag), m_t(t)
    {
      init_default_constants();
    }

    // For merkle tree size should be equal to the arity of a single hasher multiplier by sizeof(S).
    // For sponge function it could be any number.
    // Size parameter here is in bytes.
    eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const override
    {
      const unsigned arity = m_use_domain_tag ? m_t - 1 : m_t;
      bool is_sponge = false;
      int input_size_in_scalars = size / sizeof(S);
      if ((config.batch == 1) && (input_size_in_scalars != (m_use_domain_tag ? m_t - 1 : m_t))) { // Check if sponge
                                                                                                  // function.
        is_sponge = true;
        if (config.batch != 1) {
          ICICLE_LOG_ERROR << "The only supported value of config.batch for sponge functions is 1.\n";
          return eIcicleError::INVALID_ARGUMENT;
        }
      } // sponge function
      else { // Non-sponge function.
        if ((m_use_domain_tag ? input_size_in_scalars : input_size_in_scalars - 1) % (m_t - 1) != 0) {
          ICICLE_LOG_ERROR << "Padding isn't supported for non-sponge function hash. The following should be true: "
                              "((m_use_domain_tag ? size : size-1) % (m_t-1) != 0).\n";
          return eIcicleError::INVALID_ARGUMENT;
        }
      } // Non-sponge function.

      const unsigned int T = m_t;
      bool is_unsupported_T_for_this_field = poseidon2_constants[T].nof_upper_full_rounds == 0;
      if (is_unsupported_T_for_this_field) {
        ICICLE_LOG_ERROR << "Unsupported poseidon width (t = " << T << ") for this field! Planned for next version";
        return eIcicleError::API_NOT_IMPLEMENTED;
      }

      int alpha = poseidon2_constants[T].alpha;
      int nof_upper_full_rounds = poseidon2_constants[T].nof_upper_full_rounds;
      int nof_partial_rounds = poseidon2_constants[T].nof_partial_rounds;
      int nof_bottom_full_rounds = poseidon2_constants[T].nof_bottom_full_rounds;
      S* rounds_constants = poseidon2_constants[T].rounds_constants;
      S* mds_matrix = poseidon2_constants[T].mds_matrix;
      S* partial_matrix_diagonal_m1 = poseidon2_constants[T].partial_matrix_diagonal_m1;

      // Allocate temporary memory for intermediate calcs and in order not to change the input.
      // int sponge_nof_hashers = m_use_domain_tag ? (input_size_in_scalars / arity) :  ((input_size_in_scalars - 1) /
      // (arity - 1)); int tmp_fields_nof_scalars = is_sponge ? (T * sponge_nof_hashers) : (T * config.batch); S*
      // tmp_fields = new S[tmp_fields_nof_scalars];
      S* tmp_fields;
      S* tmp_fields_init_ptr; // This pointer to keep initial tmp_fields value to perform a easy rollback when needed.
      int sponge_nof_hashers;
      const S* in_fields = (S*)(input);
      int padding_size = 0;
      S* padding;
      if (is_sponge) {
        if (input_size_in_scalars < T) { // Single hasher in the chain.
          sponge_nof_hashers = 1;
          padding_size = T - (input_size_in_scalars + (m_use_domain_tag == true));
        } else if (input_size_in_scalars >= T) { // More than a single hasher in the chain.
          sponge_nof_hashers = (input_size_in_scalars - !(m_use_domain_tag == true) + (T - 2)) / (T - 1);
          bool is_padding_needed = (input_size_in_scalars - !(m_use_domain_tag == true)) % (T - 1);
          if (is_padding_needed) {
            padding_size = (T - 1) - ((input_size_in_scalars - !(m_use_domain_tag == true)) % (T - 1));
          }
        }
        if (padding_size > 0) { // Fill padding array with 1,0,0,...
          padding = new S[padding_size];
          padding[0] = S::from(1);
          for (int i = 1; i < padding_size; i++) {
            padding[i] = S::from(0);
          }
        }
        tmp_fields = new S[T * sponge_nof_hashers];
        tmp_fields_init_ptr = tmp_fields;
        // Take care of hasher 0. It's done separately of the rest of the hashers because of the domain tag.
        if (m_use_domain_tag) {
          // Domain tag exists only for the first hasher. For the rest of the hashers this
          // input is undefined at this stage and its value will be set later.
          // tmp_fields = {{dt, in0}, {undef, in1}, {undef, in2}, etc.}
          memcpy(tmp_fields, &m_domain_tag, sizeof(S));
        } else {
          // tmp_fields = {{in0 (T inputs)}, {undef, in1 (T-1 inputs)}, {under, in2 (T-1 inputs)}, etc.}
          memcpy(tmp_fields, &in_fields[0], sizeof(S));
          in_fields += 1;
        }
        tmp_fields += 1;
        // Take care of rest of the hashers (T-1 scalar to each hasher).
        for (int hasher_idx = 0; hasher_idx < sponge_nof_hashers; hasher_idx++) {
          if (hasher_idx == sponge_nof_hashers - 1 && padding_size > 0) {
            // Last hasher in the chain. Take care of padding if needed.
            memcpy(tmp_fields, in_fields, (T - padding_size - 1) * sizeof(S));
            memcpy(tmp_fields + T - padding_size - 1, padding, padding_size * sizeof(S));
          } else { // Not a last hasher in the chain. There is no padding.
            memcpy(tmp_fields, in_fields, (T - 1) * sizeof(S));
          }
          in_fields += (T - 1);
          tmp_fields += T;
        }
        tmp_fields = tmp_fields_init_ptr; // Rollback to initial value.
      } // if (is_sponge) {
      else { // Not a sponge function. The is no padding.
        // Input of each hash should have domain tag at its input.
        // tmp_fields = {{dt, in0 (T-1 inputs)}, {dt, in1 (T-1 inputs)}, {dt, in2 (T-1 inputs)}, etc.}
        tmp_fields = new S[T * config.batch];
        tmp_fields_init_ptr = tmp_fields; // Keep tmp_fields pointer for delete.
        if (m_use_domain_tag) {
          for (int batch_idx = 0; batch_idx < config.batch; batch_idx++) {
            memcpy(tmp_fields, &m_domain_tag, sizeof(S));
            memcpy(tmp_fields + 1, in_fields, (T - 1) * sizeof(S));
            in_fields += (T - 1);
            tmp_fields += T;
          }
          tmp_fields = tmp_fields_init_ptr; // Rollback to initial value.
        } else {
          // tmp_fields = {{in0 (T inputs)}, {in1 (T inputs)}, {in2 (T inputs)}, etc.}
          memcpy(tmp_fields, in_fields, T * config.batch * sizeof(S));
        }
      }

      // Hashes processing.
      if (is_sponge) {
        // Call hash_single for hasher[0]
        eIcicleError err = hash_single(
          tmp_fields /* input */, tmp_fields /* output */, alpha, nof_upper_full_rounds, nof_partial_rounds,
          nof_bottom_full_rounds, rounds_constants, mds_matrix, partial_matrix_diagonal_m1);
        S* tmp_fields_tmp_ptr = tmp_fields; // Save current pointer in order to access prev output.
        if (err != eIcicleError::SUCCESS) return err;
        if (sponge_nof_hashers != 1) {
          tmp_fields[T] = tmp_fields[0]; // Current first output is an input to the next hasher.
        }
        tmp_fields += T;
        // Process rest of the hashers.
        for (int hasher_idx = 1; hasher_idx < sponge_nof_hashers; hasher_idx++) {
          // The first output of the prev hasher is the first input of the current hasher.
          // The T-1 new inputs of the current hasher should be added to the T-1 outputs of the
          // prev hasher (starting from index 1).
          for (int i = 1; i < T; i++) {
            tmp_fields[i] = tmp_fields_tmp_ptr[i] + tmp_fields[i];
          }
          eIcicleError err = hash_single(
            tmp_fields /* input */, tmp_fields /* output */, alpha, nof_upper_full_rounds, nof_partial_rounds,
            nof_bottom_full_rounds, rounds_constants, mds_matrix, partial_matrix_diagonal_m1);
          tmp_fields_tmp_ptr = tmp_fields; // Save current pointer in order to access prev output.
          if (err != eIcicleError::SUCCESS) return err;
          if (hasher_idx != sponge_nof_hashers - 1) // Not to do in the last loop to prevent mem leak.
            tmp_fields[T] = tmp_fields[0];          // Fill first scalar of the input to the next hasher.
          tmp_fields += T; // Now tmp_fields points to input of the next hasher before the addition.
        } // for (int hasher_idx = 1; hasher_idx < sponge_nof_hashers; hasher_idx++) {
        tmp_fields -= T; // Rollback to the last hasher output.
        memcpy(output, (std::byte*)(&tmp_fields[1]), sizeof(S));
        tmp_fields = tmp_fields_init_ptr; // Rollback to initial value.
      } else {                            // Not a sponge function.
        for (int batch_hash_idx = 0; batch_hash_idx < config.batch; batch_hash_idx++) {
          eIcicleError err = hash_single(
            tmp_fields /* input */, tmp_fields /* output */, alpha, nof_upper_full_rounds, nof_partial_rounds,
            nof_bottom_full_rounds, rounds_constants, mds_matrix, partial_matrix_diagonal_m1);
          if (err != eIcicleError::SUCCESS) return err;
          memcpy(output, (std::byte*)(&tmp_fields[1]), sizeof(S));
          tmp_fields += T;
          output += sizeof(S);
        }
        tmp_fields = tmp_fields_init_ptr; // Rollback to initial value.
      }

      delete[] tmp_fields;
      if (padding_size != 0) delete[] padding;
      tmp_fields = nullptr;

      return eIcicleError::SUCCESS;
    } // eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const
      // override

  private:
    // // DEBUG start. Do not remove!!!
    // void print_state(std::string str, const S* state_to_print, int count)  const {
    //   std::cout << str << std::endl;
    //   for (int state_idx = 0; state_idx < count; state_idx++) { // Columns of matrix.
    //     std::cout << std::hex << state_to_print[state_idx] << std::endl;
    //   }
    // }
    // void print_matrix(std::string str, S* matrix_to_print)  const {
    //   std::cout << str << std::endl;
    //   unsigned int T = m_t;
    //   for (int matrix_idx = 0; matrix_idx < T*T; matrix_idx++) { // Columns of matrix.
    //     std::cout << std::hex << matrix_to_print[matrix_idx] << std::endl;
    //   }
    // }
    // // DEBUG end

    // This function performs a single hash according to parameters in the poseidon2_constants[] struct.
    // eIcicleError hash_single(const std::byte* input, std::byte* output) const
    eIcicleError hash_single(
      S* tmp_fields,
      S* hasher_output,
      int alpha,
      int nof_upper_full_rounds,
      int nof_partial_rounds,
      int nof_bottom_full_rounds,
      S* rounds_constants,
      S* mds_matrix,
      S* partial_matrix_diagonal_m1) const
    {
      const unsigned int T = m_t;

      // Pre-rounds full matrix multiplication.
      full_matrix_mul_by_vector(tmp_fields, mds_matrix, tmp_fields);

      // Upper full rounds.
      full_rounds(nof_upper_full_rounds, tmp_fields, rounds_constants);

      // Partial rounds. Perform calculation only for the first element of *tmp_fields.
      for (int partial_rounds_idx = 0; partial_rounds_idx < nof_partial_rounds; partial_rounds_idx++) {
        // Add round constants
        tmp_fields[0] = tmp_fields[0] + *rounds_constants++;
        // S box
        tmp_fields[0] = S::pow(tmp_fields[0], alpha);
        // Multiplication by partial (sparse) matrix.
        partial_matrix_diagonal_m1_mul_by_vector(tmp_fields, partial_matrix_diagonal_m1, tmp_fields);
      }

      // Bottom full rounds.
      full_rounds(nof_bottom_full_rounds, tmp_fields, rounds_constants);

      memcpy(hasher_output, (std::byte*)(tmp_fields), T * sizeof(S));
      // memcpy(output, (std::byte*)(&tmp_fields[1]), sizeof(S));

      return eIcicleError::SUCCESS;
    } // eIcicleError hash_single(const std::byte* input, std::byte* output) const

    // This function performs a partial_matrix_diagonal_m1 matrix by vector multiplication.
    // Note that in order to increase the performance the partial matrix diagonal values
    // are actually partial matrix diagonal values minus 1.
    void partial_matrix_diagonal_m1_mul_by_vector(const S* vec_in, const S* matrix_in, S* result) const
    {
      unsigned int T = m_t;
      S tmp_col_res[T]; // Have to use temp storage because vec_in and result are the same storage.
      S vec_in_sum = vec_in[0];
      for (int vec_in_idx = 1; vec_in_idx < T; vec_in_idx++)
        vec_in_sum = vec_in_sum + vec_in[vec_in_idx];
      for (int col_idx = 0; col_idx < T; col_idx++)
        tmp_col_res[col_idx] = vec_in_sum + matrix_in[col_idx] * vec_in[col_idx];
      for (int col_idx = 0; col_idx < T; col_idx++) { // This copy is needed because vec_in and result storages are
                                                      // actually the same storage when calling to the function.
        result[col_idx] = tmp_col_res[col_idx];
      }
    } // void partial_matrix_diagonal_m1_mul_by_vector(const S* vec_in, const S* matrix_in, S* result) const

    // This function performs a full matrix by vector multiplication.
    void full_matrix_mul_by_vector(const S* vec_in, const S* matrix_in, S* result) const
    {
      unsigned int T = m_t;
      S tmp_col_res[T]; // Have to use temp storage because vec_in and result are the same storage.
      for (int row_idx = 0; row_idx < T; row_idx++) { // Rows of matrix.
        tmp_col_res[row_idx] = matrix_in[row_idx * T] * vec_in[0];
        for (int col_idx = 1; col_idx < T; col_idx++) { // Columns of matrix.
          tmp_col_res[row_idx] = tmp_col_res[row_idx] + matrix_in[row_idx * T + col_idx] * vec_in[col_idx];
        }
      }
      for (int col_idx = 0; col_idx < T; col_idx++) { // This copy is needed because vec_in and result storages are
                                                      // actually the same storage when calling to the function.
        result[col_idx] = tmp_col_res[col_idx];
      }
    } // eIcicleError hash_single(const std::byte* input, std::byte* output) const

    // This function performs a needed number of full rounds calculations.
    void full_rounds(const unsigned int nof_full_rounds, S* in_out_fields, S*& rounds_constants) const
    {
      unsigned int T = m_t;
      unsigned int alpha = poseidon2_constants[T].alpha;
      S* mds_matrix = poseidon2_constants[T].mds_matrix;
      for (int full_rounds_idx = 0; full_rounds_idx < nof_full_rounds; full_rounds_idx++) {
        for (int state_idx = 0; state_idx < T; state_idx++) {
          // Add round constants
          in_out_fields[state_idx] = in_out_fields[state_idx] + *rounds_constants++;
        }
        for (int state_idx = 0; state_idx < T; state_idx++) {
          // S box
          in_out_fields[state_idx] = S::pow(in_out_fields[state_idx], alpha);
        }
        // Multiplication by matrix
        full_matrix_mul_by_vector(in_out_fields, mds_matrix, in_out_fields);
      }
    } // void full_rounds(const unsigned int nof_full_rounds, S* in_out_fields, S*& rounds_constants) const

    const bool m_use_domain_tag;
    const S m_domain_tag;
    const unsigned int m_t;
  }; // class Poseidon2BackendCPU : public HashBackend

  static eIcicleError create_cpu_poseidon2_hash_backend(
    const Device& device, unsigned t, const scalar_t* domain_tag, std::shared_ptr<HashBackend>& backend /*OUT*/)
  {
    backend = std::make_shared<Poseidon2BackendCPU<scalar_t>>(t, domain_tag);
    return eIcicleError::SUCCESS;
  }

  REGISTER_CREATE_POSEIDON2_BACKEND("CPU", create_cpu_poseidon2_hash_backend);

} // namespace icicle