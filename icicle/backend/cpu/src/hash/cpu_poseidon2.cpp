#include "icicle/backend/hash/poseidon2_backend.h"
#include "icicle/utils/utils.h"
#include "icicle/fields/field.h"
#include <vector>

#if FIELD_ID == BN254
  #include "icicle/hash/poseidon2_constants/constants/bn254_poseidon2.h"
// using namespace poseidon2_constants_bn254;
// #elif FIELD_ID == BLS12_381
//   #include "icicle/hash/poseidon2_constants/constants/bls12_381_poseidon2.h"
// using namespace poseidon2_constants_bls12_381;
// #elif FIELD_ID == BLS12_377
//   #include "icicle/hash/poseidon2_constants/constants/bls12_377_poseidon2.h"
// using namespace poseidon2_constants_bls12_377;
// #elif FIELD_ID == BW6_761
//   #include "icicle/hash/poseidon2_constants/constants/bw6_761_poseidon2.h"
// using namespace poseidon2_constants_bw6_761;
// #elif FIELD_ID == GRUMPKIN
//   #include "icicle/hash/poseidon2_constants/constants/grumpkin_poseidon2.h"
// using namespace poseidon2_constants_grumpkin;
// #elif FIELD_ID == M31
//   #include "icicle/hash/poseidon2_constants/constants/m31_poseidon2.h"
// using namespace poseidon2_constants_m31;
// #elif FIELD_ID == BABY_BEAR
//   #include "icicle/hash/poseidon2_constants/constants/babybear_poseidon2.h"
// using namespace poseidon2_constants_babybear;
// #elif FIELD_ID == STARK_252
//   #include "icicle/hash/poseidon2_constants/constants/stark252_poseidon2.h"
// using namespace poseidon2_constants_stark252;
#endif

namespace icicle {

#define POSEIDON2_MAX_t 24

  static unsigned int poseidon2_legal_width[] = {2, 3, 4, 8, 12, 16, 20, 24};

  static Poseidon2ConstantsOptions<scalar_t>
    poseidon2_constants[POSEIDON2_MAX_t + 1]; // The size of this array is POSEIDON2_MAX_t + 1 because Poseidon2 max t is 24. Only
                                            // terms in poseidon2_legal_width are filled with data. Rest of the terms are not
                                            // relevant.
  static bool s_cpu_backend_poseidon2_constants_initialized = false;

  static eIcicleError init_default_constants()
  {
    // if (s_cpu_backend_poseidon2_constants_initialized) { return eIcicleError::SUCCESS; }

    // unsigned int alpha;
    // unsigned int partial_rounds;
    // unsigned int upper_full_rounds;
    // unsigned int bottom_full_rounds;
    // // unsigned char* constants;
    // std::string* constants;
    // // At this stage it's still unknown what t and use_domain_tag will be used.
    // // That's the reason that all the relevant members of the poseidon2_constants array are
    // // loaded at this stage.
    // for (int t_idx = 0; t_idx < std::size(poseidon2_legal_width); t_idx++) {
    //   unsigned int T = poseidon2_legal_width[t_idx]; // Single poseidon2 hash width
    //   switch (T) {
    //     case 2:
    //       alpha = alpha_2;
    //       constants = round_constants_2;
    //       partial_rounds = partial_rounds_2;
    //       upper_full_rounds = half_full_rounds_2;
    //       bottom_full_rounds = half_full_rounds_2;
    //       break;
    //     case 3:
    //       alpha = alpha_3;
    //       constants = round_constants_3;
    //       partial_rounds = partial_rounds_3;
    //       upper_full_rounds = half_full_rounds_3;
    //       bottom_full_rounds = half_full_rounds_3;
    //       break;
    //     case 4:
    //       constants = round_constants_4;
    //       partial_rounds = partial_rounds_4;
    //       upper_full_rounds = half_full_rounds_4;
    //       bottom_full_rounds = half_full_rounds_4;
    //       break;
    //     case 8:
    //       constants = round_constants_8;
    //       partial_rounds = partial_rounds_8;
    //       upper_full_rounds = half_full_rounds_8;
    //       bottom_full_rounds = half_full_rounds_8;
    //       break;
    //     case 12:
    //       constants = round_constants_12;
    //       partial_rounds = partial_rounds_12;
    //       upper_full_rounds = half_full_rounds_12;
    //       bottom_full_rounds = half_full_rounds_12;
    //       break;
    //     case 16:
    //       constants = round_constants_16;
    //       partial_rounds = partial_rounds_16;
    //       upper_full_rounds = half_full_rounds_16;
    //       bottom_full_rounds = half_full_rounds_16;
    //       break;
    //     case 20:
    //       constants = round_constants_20;
    //       partial_rounds = partial_rounds_20;
    //       upper_full_rounds = half_full_rounds_20;
    //       bottom_full_rounds = half_full_rounds_20;
    //       break;
    //     case 24:
    //       constants = round_constants_24;
    //       partial_rounds = partial_rounds_24;
    //       upper_full_rounds = half_full_rounds_24;
    //       bottom_full_rounds = half_full_rounds_24;
    //       break;
    //   default:
    //     ICICLE_LOG_ERROR << "cpu_poseidon2_init_default_constants: T (width) must be one of [3, 5, 9, 12]";
    //     return eIcicleError::INVALID_ARGUMENT;
    //   }
    //   scalar_t* h_constants = reinterpret_cast<scalar_t*>(constants);

    //   poseidon2_constants[T].t = T;
    //   poseidon2_constants[T].alpha = 5;
    //   poseidon2_constants[T].nof_upper_full_rounds = upper_full_rounds;
    //   poseidon2_constants[T].nof_bottom_full_rounds = bottom_full_rounds;
    //   poseidon2_constants[T].nof_partial_rounds = partial_rounds;
    //   unsigned int round_constants_len =
    //     T * (poseidon2_constants[T].nof_upper_full_rounds + poseidon2_constants[T].nof_bottom_full_rounds) +
    //     partial_rounds;
    //   unsigned int mds_matrix_len = T * T;
    //   poseidon2_constants[T].rounds_constants = h_constants;
    //   poseidon2_constants[T].mds_matrix = poseidon2_constants[T].rounds_constants + round_constants_len;
    //   poseidon2_constants[T].pre_matrix = poseidon2_constants[T].mds_matrix + mds_matrix_len;
    //   poseidon2_constants[T].sparse_matrices =
    //     poseidon2_constants[T].pre_matrix + mds_matrix_len; // pre_matrix and mds_matrix have the same length.
    // }

    // s_cpu_backend_poseidon2_constants_initialized = true;
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
      // init_default_constants();
    }

    // For merkle tree size should be equal to the arity of a single hasher multiplier by sizeof(S).
    // For sponge function it could be any number.
    eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const override
    {
      // unsigned int arity = m_use_domain_tag ? m_t - 1 : m_t;

      // // Currently sponge and padding functionalities are not supported.
      // if (size != arity * sizeof(S)) {
      //   ICICLE_LOG_ERROR
      //     << "Sponge function still isn't supported. The following should be true: (size == T) but it is not.\n";
      //   return eIcicleError::INVALID_ARGUMENT;
      // }

      // // Call hash_single config.batch times.
      // for (int batch_hash_idx = 0; batch_hash_idx < config.batch; batch_hash_idx++) {
      //   hash_single(input, output);
      //   input += arity * sizeof(S);
      //   output += sizeof(S);
      // }

      return eIcicleError::SUCCESS;
    }

  private:
    // This function performs a single hash according to parameters in the poseidon2_constants[] struct.
    eIcicleError hash_single(const std::byte* input, std::byte* output) const
    {
    //   const unsigned int T = m_t;

    //   unsigned int alpha = poseidon2_constants[T].alpha;
    //   unsigned int nof_upper_full_rounds = poseidon2_constants[T].nof_upper_full_rounds;
    //   unsigned int nof_partial_rounds = poseidon2_constants[T].nof_partial_rounds;
    //   unsigned int nof_bottom_full_rounds = poseidon2_constants[T].nof_bottom_full_rounds;
    //   S* rounds_constants = poseidon2_constants[T].rounds_constants;
    //   S* mds_matrix = poseidon2_constants[T].mds_matrix;
    //   S* pre_matrix = poseidon2_constants[T].pre_matrix;
    //   S* sparse_matrices = poseidon2_constants[T].sparse_matrices;
    //   // Allocate temporary memory for intermediate calcs.
    //   S* tmp_fields = new S[T];
    //   // Casting from bytes to scalar.
    //   const S* in_fields = (S*)(input);
    //   // Copy input scalar to the output (as a temp storage) to be used in the rounds.
    //   // *tmp_fields are used as a temp storage during the calculations in this function.
    //   if (m_use_domain_tag) {
    //     // in that case we hash [domain_tag, t-1 field elements]
    //     memcpy(tmp_fields, &m_domain_tag, sizeof(S));
    //     memcpy(tmp_fields + 1, in_fields, (T - 1) * sizeof(S));
    //   } else {
    //     // in that case we hash [t field elements]
    //     memcpy(tmp_fields, in_fields, T * sizeof(S));
    //   }

    //   // Add pre-round constants.
    //   for (int state_idx = 0; state_idx < T; state_idx++) {
    //     tmp_fields[state_idx] = tmp_fields[state_idx] + *rounds_constants++;
    //   }

    //   // Upper full rounds.
    //   // Note that the number of full rounds is nof_full_rounds-1 because of the
    //   // pre_matrix round (last round of upper full rounds) and the last round of
    //   // the bottom fulld rounds that doesn'T have sbox operation.
    //   full_rounds(nof_upper_full_rounds - 1, tmp_fields, rounds_constants);

    //   // Single full round with pre_matrix.
    //   for (int state_idx = 0; state_idx < T; state_idx++) {
    //     // S box
    //     tmp_fields[state_idx] = S::pow(tmp_fields[state_idx], alpha);
    //   }
    //   for (int state_idx = 0; state_idx < T; state_idx++) {
    //     // Add round constants
    //     tmp_fields[state_idx] = tmp_fields[state_idx] + *rounds_constants++;
    //   }
    //   // Multiplication by pre_matrix
    //   field_vec_sqr_full_matrix_mul(tmp_fields, pre_matrix, tmp_fields);

    //   // Partial rounds. Perform calculation only for the first element of *tmp_fields.
    //   for (int partial_rounds_idx = 0; partial_rounds_idx < nof_partial_rounds; partial_rounds_idx++) {
    //     // S box
    //     tmp_fields[0] = S::pow(tmp_fields[0], alpha);
    //     // Add round constants
    //     tmp_fields[0] = tmp_fields[0] + *rounds_constants++;
    //     // Multiplication by sparse matrix.
    //     field_vec_sqr_sparse_matrix_mul(tmp_fields, &sparse_matrices[partial_rounds_idx * (2 * T - 1)], tmp_fields);
    //   }

    //   // Bottom full rounds.
    //   // Note that the number of full rounds is nof_full_rounds-1 because of the
    //   // last round implemented below.
    //   full_rounds(nof_bottom_full_rounds - 1, tmp_fields, rounds_constants);

    //   // Last full round
    //   for (int state_idx = 0; state_idx < T; state_idx++) {
    //     // S box
    //     tmp_fields[state_idx] = S::pow(tmp_fields[state_idx], alpha);
    //   }
    //   // Multiplication by MDS matrix
    //   field_vec_sqr_full_matrix_mul(tmp_fields, mds_matrix, tmp_fields);

    //   memcpy(output, (std::byte*)(&tmp_fields[1]), sizeof(S));

    //   delete[] tmp_fields;
    //   tmp_fields = nullptr;

      return eIcicleError::SUCCESS;
    }

    // This function performs a vector by sparse matrix multiplication.
    // Note that sparse_matrix is organized in memory 1st column first and then rest of the members of the 1st row.
    void field_vec_sqr_sparse_matrix_mul(const S* vec_in, const S* matrix_in, S* result) const
    {
    //   unsigned int T = m_t;
    //   S tmp_col_res[T]; // Have to use temp storage because vec_in and result are the same storage.
    //   tmp_col_res[0] = vec_in[0] * matrix_in[0];
    //   for (int col_idx = 1; col_idx < T; col_idx++) { // Calc first column result.
    //     tmp_col_res[0] = tmp_col_res[0] + vec_in[col_idx] * matrix_in[col_idx];
    //   }
    //   for (int col_idx = 1; col_idx < T; col_idx++) { // Calc rest columns results.
    //     tmp_col_res[col_idx] = vec_in[0] * matrix_in[T + col_idx - 1] + vec_in[col_idx];
    //   }
    //   for (int col_idx = 0; col_idx < T; col_idx++) { // This copy is needed because vec_in and result storages are
    //                                                   // actually the same storage when calling to the function.
    //     result[col_idx] = tmp_col_res[col_idx];
    //   }
    }

    // This function performs a vector by full matrix multiplication.
    void field_vec_sqr_full_matrix_mul(const S* vec_in, const S* matrix_in, S* result) const
    {
    //   unsigned int T = m_t;
    //   S tmp_col_res[T]; // Have to use temp storage because vec_in and result are the same storage.
    //   for (int col_idx = 0; col_idx < T; col_idx++) { // Columns of matrix.
    //     tmp_col_res[col_idx] = S::from(0);
    //     for (int row_idx = 0; row_idx < T; row_idx++) { // Matrix rows but also input vec columns.
    //       tmp_col_res[col_idx] = tmp_col_res[col_idx] + vec_in[row_idx] * matrix_in[row_idx * T + col_idx];
    //     }
    //   }
    //   for (int col_idx = 0; col_idx < T; col_idx++) { // This copy is needed because vec_in and result storages are
    //                                                   // actually the same storage when calling to the function.
    //     result[col_idx] = tmp_col_res[col_idx];
    //   }
    }   // eIcicleError hash_single(const std::byte* input, std::byte* output) const

    // This function performs a needed number of full rounds calculations.
    void full_rounds(const unsigned int nof_full_rounds, S* in_out_fields, S*& rounds_constants) const
    {
    //   unsigned int T = m_t;
    //   unsigned int alpha = poseidon2_constants[T].alpha;
    //   S* mds_matrix = poseidon2_constants[T].mds_matrix;
    //   for (int full_rounds_idx = 0; full_rounds_idx < nof_full_rounds; full_rounds_idx++) {
    //     for (int state_idx = 0; state_idx < T; state_idx++) {
    //       // S box
    //       in_out_fields[state_idx] = S::pow(in_out_fields[state_idx], alpha);
    //     }
    //     for (int state_idx = 0; state_idx < T; state_idx++) {
    //       // Add round constants
    //       in_out_fields[state_idx] = in_out_fields[state_idx] + *rounds_constants++;
    //     }
    //     // Multiplication by matrix
    //     field_vec_sqr_full_matrix_mul(in_out_fields, mds_matrix, in_out_fields);
    //   }
    }   // void full_rounds(const unsigned int nof_full_rounds, S* in_out_fields, S*& rounds_constants) const

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