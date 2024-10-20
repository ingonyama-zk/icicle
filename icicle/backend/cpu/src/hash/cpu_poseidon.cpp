#include "icicle/backend/hash/poseidon_backend.h"
#include "icicle/utils/utils.h"
#include "icicle/fields/field.h"
#include <vector>

#if FIELD_ID == BN254
  #include "poseidon/constants/bn254_poseidon.h"
using namespace poseidon_constants_bn254;
#elif FIELD_ID == BLS12_381
  #include "poseidon/constants/bls12_381_poseidon.h"
using namespace poseidon_constants_bls12_381;
#elif FIELD_ID == BLS12_377
  #include "poseidon/constants/bls12_377_poseidon.h"
using namespace poseidon_constants_bls12_377;
#elif FIELD_ID == BW6_761
  #include "poseidon/constants/bw6_761_poseidon.h"
using namespace poseidon_constants_bw6_761;
#elif FIELD_ID == GRUMPKIN
  #include "poseidon/constants/grumpkin_poseidon.h"
using namespace poseidon_constants_grumpkin;
#elif FIELD_ID == M31
  #include "poseidon/constants/m31_poseidon.h"
using namespace poseidon_constants_m31;
#elif FIELD_ID == BABY_BEAR
  #include "poseidon/constants/babybear_poseidon.h"
using namespace poseidon_constants_babybear;
#endif

// TODO Danny missing fields: babybear etc. or disable Poseidon for them?

namespace icicle {

#define POSEIDON_MAX_ARITY 12

  static unsigned int poseidon_legal_width[] = {3, 5, 9, 12};

  static PoseidonConstantsOptions<scalar_t>
    poseidon_constants[POSEIDON_MAX_ARITY + 1]; // The size of this array is 13 because Poseidon max arity is 12. Only
                                                // terms 3, 5, 9 and 12 are filled with data. Rest of the terms are not
                                                // relevant.

  static eIcicleError
  cpu_poseidon_init_constants(const Device& device, const PoseidonConstantsOptions<scalar_t>* options)
  {
    ICICLE_LOG_DEBUG << "In cpu_poseidon_init_constants() for type " << demangle<scalar_t>();
    poseidon_constants[options->arity] = *options;
    return eIcicleError::SUCCESS;
  }

  REGISTER_POSEIDON_INIT_CONSTANTS_BACKEND("CPU", cpu_poseidon_init_constants);

  static eIcicleError init_default_constants()
  {
    ICICLE_LOG_DEBUG << "In cpu_poseidon_init_default_constants() for type " << demangle<scalar_t>();
    unsigned int partial_rounds;
    unsigned char* constants;
    // At this stage it's still unknown what arity and is_domain_tag will be used.
    // That's the reason that all איק relevant members of the poseidon_constants array are
    // loaded at this stage.
    for (int arity_idx = 0; arity_idx < std::size(poseidon_legal_width); arity_idx++) {
      unsigned int T = poseidon_legal_width[arity_idx]; // Single poseidon hash width
      poseidon_constants[T].alpha = 5;
      poseidon_constants[T].nof_upper_full_rounds = 4;
      poseidon_constants[T].nof_bottom_full_rounds = 4;
      switch (T) {
      case 3:
        constants = poseidon_constants_3;
        partial_rounds = partial_rounds_3;
        break;
      case 5:
        constants = poseidon_constants_5;
        partial_rounds = partial_rounds_5;
        break;
      case 9:
        constants = poseidon_constants_9;
        partial_rounds = partial_rounds_9;
        break;
      case 12:
        constants = poseidon_constants_12;
        partial_rounds = partial_rounds_12;
        break;
      default:
        ICICLE_LOG_ERROR << "cpu_poseidon_init_default_constants: T (width) must be one of [3, 5, 9, 12]";
        return eIcicleError::INVALID_ARGUMENT;
      }
      scalar_t* h_constants = reinterpret_cast<scalar_t*>(constants);

      unsigned int round_constants_len =
        T * (poseidon_constants[T].nof_upper_full_rounds + poseidon_constants[T].nof_bottom_full_rounds) +
        partial_rounds;
      unsigned int mds_matrix_len = T * T;
      poseidon_constants[T].nof_partial_rounds = partial_rounds;
      poseidon_constants[T].rounds_constants = h_constants;
      poseidon_constants[T].mds_matrix = poseidon_constants[T].rounds_constants + round_constants_len;
      poseidon_constants[T].pre_matrix = poseidon_constants[T].mds_matrix + mds_matrix_len;
      poseidon_constants[T].sparse_matrices =
        poseidon_constants[T].pre_matrix + mds_matrix_len; // pre_matrix and mds_matrix have the same length.
    }

    return eIcicleError::SUCCESS;
  }

  template <typename S>
  class PoseidonBackendCPU : public HashBackend
  {
  public:
    PoseidonBackendCPU(
      unsigned arity, unsigned default_input_size, bool is_domain_tag, S* domain_tag_value, bool use_all_zeroes_padding)
        : HashBackend("Poseidon-CPU", sizeof(S), default_input_size)
    {
      init_default_constants();
      unsigned int width = is_domain_tag ? arity + 1 : arity;
      poseidon_constants[width].arity = arity;
      poseidon_constants[width].is_domain_tag = is_domain_tag;
      poseidon_constants[width].domain_tag_value = domain_tag_value;
      poseidon_constants[width].use_all_zeroes_padding = use_all_zeroes_padding;
      // These values should be kept as a class members because otherwise the hash width will be unknown
      // in the hash function.
      m_arity = arity;
      m_is_domain_tag = is_domain_tag;
    }

    eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const override
    {
      ICICLE_LOG_DEBUG << "Poseidon CPU hash() " << size << " bytes, for type " << demangle<S>()
                       << ", batch=" << config.batch;

      unsigned int T = m_is_domain_tag ? m_arity + 1 : m_arity;

      // Currently sponge and padding functionalities are not supported.
      // Therefore check that size/T == config.batch
      ICICLE_ASSERT(size / (T * sizeof(S)) == sizeof(S) * config.batch)
        << "Sponge function still isn't supported. The following should be true: size/T == config.batch but it is "
           "not.\n";

      // Call hash_single config.batch times.
      for (int batch_hash_idx = 0; batch_hash_idx < config.batch; batch_hash_idx++) {
        hash_single(input, size, output);
        input += m_arity * sizeof(S);
        output += sizeof(S);
      }

      return eIcicleError::SUCCESS;
    }

    eIcicleError hash_single(const std::byte* input, uint64_t size, std::byte* output) const
    {
      ICICLE_LOG_DEBUG << "Poseidon CPU hash_single() " << size << " bytes, for type " << demangle<S>();

      unsigned int T = m_is_domain_tag ? m_arity + 1 : m_arity;

      unsigned int alpha = poseidon_constants[T].alpha;
      unsigned int nof_upper_full_rounds = poseidon_constants[T].nof_upper_full_rounds;
      unsigned int nof_partial_rounds = poseidon_constants[T].nof_partial_rounds;
      unsigned int nof_bottom_full_rounds = poseidon_constants[T].nof_bottom_full_rounds;
      S* rounds_constants = poseidon_constants[T].rounds_constants;
      S* mds_matrix = poseidon_constants[T].mds_matrix;
      S* pre_matrix = poseidon_constants[T].pre_matrix;
      S* sparse_matrices = poseidon_constants[T].sparse_matrices;
      // Allocate temporary memory for intermediate calcs.
      S* tmp_fields = new S[T];
      // Casting from bytes to scalar.
      const S* in_fields = (S*)(input);
      // Copy input scalar to the output (as a temp storage) to be used in the rounds.
      // *tmp_fields are used as a temp storage during the calculations in this function.
      memcpy(tmp_fields, in_fields, T * sizeof(S));

      // Add pre-round constants.
      for (int state_idx = 0; state_idx < T; state_idx++) {
        tmp_fields[state_idx] = tmp_fields[state_idx] + *rounds_constants++;
      }

      // Upper full rounds.
      full_rounds(nof_upper_full_rounds, tmp_fields, rounds_constants);

      // Single full round with pre_matrix.
      for (int state_idx = 0; state_idx < T; state_idx++) {
        // S box
        tmp_fields[state_idx] = S::pow(tmp_fields[state_idx], alpha);
      }
      for (int state_idx = 0; state_idx < T; state_idx++) {
        // Add round constants
        tmp_fields[state_idx] = tmp_fields[state_idx] + *rounds_constants++;
      }
      // Multiplication by pre_matrix
      field_vec_sqr_full_matrix_mul(tmp_fields, pre_matrix, tmp_fields);

      // Partial rounds. Perform calculation only for the first element of *tmp_fields.
      for (int partial_rounds_idx = 0; partial_rounds_idx < nof_partial_rounds; partial_rounds_idx++) {
        // S box
        tmp_fields[0] = S::pow(tmp_fields[0], alpha);
        // Add round constants
        tmp_fields[0] = tmp_fields[0] + *rounds_constants++;
        // Multiplication by sparse matrix.
        field_vec_sqr_sparse_matrix_mul(tmp_fields, &sparse_matrices[partial_rounds_idx * (2 * T - 1)], tmp_fields);
      }

      // Bottom full rounds.
      full_rounds(nof_bottom_full_rounds, tmp_fields, rounds_constants);

      // Last full round
      for (int state_idx = 0; state_idx < T; state_idx++) {
        // S box
        tmp_fields[state_idx] = S::pow(tmp_fields[state_idx], alpha);
      }
      // Multiplication by MDS matrix
      field_vec_sqr_full_matrix_mul(tmp_fields, mds_matrix, tmp_fields);

      memcpy(output, (std::byte*)(&tmp_fields[1]), sizeof(S));

      delete[] tmp_fields;
      tmp_fields = nullptr;

      return eIcicleError::SUCCESS;
    }

    void field_vec_sqr_sparse_matrix_mul(const S* vec_in, const S* matrix_in, S* result) const
    {
      unsigned int T = m_is_domain_tag ? m_arity + 1 : m_arity;
      S tmp_col_res[T]; // Have to use temp storage because vec_in and result are the same storage.
      // Note that sparse_matrix is organized in memory 1st column first and then rest of the members of the 1st row.
      tmp_col_res[0] = vec_in[0] * matrix_in[0];
      for (int col_idx = 1; col_idx < T; col_idx++) { // Calc first column result.
        tmp_col_res[0] = tmp_col_res[0] + vec_in[col_idx] * matrix_in[col_idx];
      }
      for (int col_idx = 1; col_idx < T; col_idx++) { // Calc rest columns results.
        tmp_col_res[col_idx] = vec_in[0] * matrix_in[T + col_idx - 1] + vec_in[col_idx];
      }
      for (int col_idx = 0; col_idx < T; col_idx++) { // This copy is needed because vec_in and result storages are
                                                      // actually the same storage when calling to the function.
        result[col_idx] = tmp_col_res[col_idx];
      }
    }

    void field_vec_sqr_full_matrix_mul(const S* vec_in, const S* matrix_in, S* result) const
    {
      unsigned int T = m_is_domain_tag ? m_arity + 1 : m_arity;
      S tmp_col_res[T]; // Have to use temp storage because vec_in and result are the same storage.
      for (int col_idx = 0; col_idx < T; col_idx++) { // Columns of matrix.
        tmp_col_res[col_idx] = S::from(0);
        for (int row_idx = 0; row_idx < T; row_idx++) { // Matrix rows but also input vec columns.
          tmp_col_res[col_idx] = tmp_col_res[col_idx] + vec_in[row_idx] * matrix_in[row_idx * T + col_idx];
        }
      }
      for (int col_idx = 0; col_idx < T; col_idx++) { // This copy is needed because vec_in and result storages are
                                                      // actually the same storage when calling to the function.
        result[col_idx] = tmp_col_res[col_idx];
      }
    }

    void full_rounds(const unsigned int nof_full_rounds, S* in_out_fields, /* const */ S*& rounds_constants) const
    {
      unsigned int T = m_is_domain_tag ? m_arity + 1 : m_arity;
      unsigned int alpha = poseidon_constants[T].alpha;
      S* mds_matrix = poseidon_constants[T].mds_matrix;
      for (int full_rounds_idx = 0; full_rounds_idx < nof_full_rounds - 1;
           full_rounds_idx++) { // Note that the number of full rounds is nof_full_rounds-1 because of the
                                // pre_matrix round (last round of upper full rounds) and the last round of
                                // the bottom fulld rounds that doesn'T have sbox operation.
        for (int state_idx = 0; state_idx < T; state_idx++) {
          // S box
          in_out_fields[state_idx] = S::pow(in_out_fields[state_idx], alpha);
        }
        for (int state_idx = 0; state_idx < T; state_idx++) {
          // Add round constants
          in_out_fields[state_idx] = in_out_fields[state_idx] + *rounds_constants++;
        }
        // Multiplication by matrix
        field_vec_sqr_full_matrix_mul(in_out_fields, mds_matrix, in_out_fields);
      }
    }

  private:
    bool m_is_domain_tag;
    unsigned int m_arity;
  }; // class PoseidonBackendCPU : public HashBackend

  static eIcicleError create_cpu_poseidon_hash_backend(
    const Device& device,
    unsigned arity,
    unsigned default_input_size,
    bool is_domain_tag,
    scalar_t* domain_tag_value,
    bool use_all_zeroes_padding,
    std::shared_ptr<HashBackend>& backend /*OUT*/,
    const scalar_t& phantom)
  {
    ICICLE_LOG_DEBUG << "in create_cpu_poseidon_hash_backend(arity=" << arity << ")";
    backend = std::make_shared<PoseidonBackendCPU<scalar_t>>(
      arity, default_input_size, is_domain_tag, domain_tag_value, use_all_zeroes_padding);
    return eIcicleError::SUCCESS;
  }

  REGISTER_CREATE_POSEIDON_BACKEND("CPU", create_cpu_poseidon_hash_backend);

} // namespace icicle