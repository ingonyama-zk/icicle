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
#endif

namespace icicle {

  #define POSEIDON_MAX_ARITY 12

  static unsigned int poseidon_legal_arities[] = {3, 5, 9, 12};

  static PoseidonConstantsInitOptions<scalar_t> poseidon_constants[POSEIDON_MAX_ARITY+1];   // The size of this array is 13 because Poseidon max arity is 12. Only terms 3, 5, 9 and 12 are filled with data. Rest of the terms are not relevant.

  static eIcicleError
  cpu_poseidon_init_constants(const Device& device, const PoseidonConstantsInitOptions<scalar_t>* options)
  {
    ICICLE_LOG_DEBUG << "in cpu_poseidon_init_constants() for type " << demangle<scalar_t>();
    poseidon_constants[options->arity] = *options;
    return eIcicleError::SUCCESS;
  }

  REGISTER_POSEIDON_INIT_CONSTANTS_BACKEND("CPU", cpu_poseidon_init_constants);

  // // static eIcicleError cpu_poseidon_init_default_constants(const Device& device, const scalar_t& phantom)   DANNY
  // static eIcicleError cpu_poseidon_init_default_constants(const Device& device, /* unsigned int arity, */ const scalar_t& phantom)
  // {
  //   ICICLE_LOG_DEBUG << "in cpu_poseidon_init_default_constants() for type " << demangle<scalar_t>();
  //   unsigned int partial_rounds;
  //   unsigned char* constants;
  //   for (int arity_idx = 0; arity_idx < std::size(poseidon_legal_arities); arity_idx++) {
  //     unsigned int arity = poseidon_legal_arities[arity_idx];
  //     poseidon_constants[arity].alpha = 5;
  //     poseidon_constants[arity].nof_upper_full_rounds = 4;
  //     poseidon_constants[arity].nof_end_full_rounds = 4;      
  //     switch (poseidon_legal_arities[arity_idx]) {
  //       case 3:
  //         constants = poseidon_constants_3;
  //         partial_rounds = partial_rounds_3;
  //         break;
  //       case 5:
  //         constants = poseidon_constants_5;
  //         partial_rounds = partial_rounds_5;
  //         break;
  //       case 9:
  //         constants = poseidon_constants_9;
  //         partial_rounds = partial_rounds_9;
  //         break;
  //       case 12:
  //         constants = poseidon_constants_12;
  //         partial_rounds = partial_rounds_12;
  //         break;
  //       default:
  //         ICICLE_LOG_ERROR << "cpu_poseidon_init_default_constants: #arity must be one of [2, 4, 8, 11]";
  //         return eIcicleError::INVALID_ARGUMENT;
  //     }
  //   }

  //   scalar_t* h_constants = reinterpret_cast<scalar_t*>(constants);

  //   for (int arity_idx = 0; arity_idx < std::size(poseidon_legal_arities); arity_idx++) {
  //     unsigned int arity = poseidon_legal_arities[arity_idx];
  //     unsigned int round_constants_len = arity * (poseidon_constants[arity].nof_upper_full_rounds + poseidon_constants[arity].nof_upper_full_rounds) + partial_rounds;
  //     unsigned int mds_matrix_len = arity * arity;      
  //     poseidon_constants[arity].rounds_constants = h_constants;
  //     poseidon_constants[arity].mds_matrix = poseidon_constants[arity].rounds_constants + round_constants_len;
  //     poseidon_constants[arity].pre_matrix = poseidon_constants[arity].mds_matrix + mds_matrix_len;
  //     poseidon_constants[arity].sparse_matrices = poseidon_constants[arity].pre_matrix + mds_matrix_len;
  //   }
  //   return eIcicleError::SUCCESS;
  // }

  // REGISTER_POSEIDON_INIT_DEFAULT_CONSTANTS_BACKEND("CPU", cpu_poseidon_init_default_constants);

  // DEBUG
  int print_input_bytes(std::string input_type, const std::byte* input, uint64_t size) {
    int nof_lines = size / 16;
    for (int line_idx = 0; line_idx < nof_lines; line_idx++) {
      for (int byte_idx = 0; byte_idx < 16; byte_idx++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(*input) << " ";
        input++;
        std::cout << std::endl;
      }
    }
    return 0;
  }
  // DEBUG

  template <typename S>
  class PoseidonBackendCPU : public HashBackend
  {
  public:
    PoseidonBackendCPU(unsigned arity) : HashBackend("Poseidon-CPU", sizeof(S), arity * sizeof(S)) {
      cpu_poseidon_init_default_constants("Poseidon-CPU", scalar_t::from(0));
    }

  eIcicleError cpu_poseidon_init_default_constants(const Device& device, const scalar_t& phantom)
  {
    ICICLE_LOG_DEBUG << "in cpu_poseidon_init_default_constants() for type " << demangle<scalar_t>();
    unsigned int partial_rounds;
    unsigned char* constants;
    for (int arity_idx = 0; arity_idx < std::size(poseidon_legal_arities); arity_idx++) {
      unsigned int arity = poseidon_legal_arities[arity_idx];
      poseidon_constants[arity].alpha = 5;
      poseidon_constants[arity].nof_upper_full_rounds = 4;
      poseidon_constants[arity].nof_end_full_rounds = 4;      
      switch (poseidon_legal_arities[arity_idx]) {
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
          ICICLE_LOG_ERROR << "cpu_poseidon_init_default_constants: #arity must be one of [2, 4, 8, 11]";
          return eIcicleError::INVALID_ARGUMENT;
      }
    }

    scalar_t* h_constants = reinterpret_cast<scalar_t*>(constants);

    for (int arity_idx = 0; arity_idx < std::size(poseidon_legal_arities); arity_idx++) {
      unsigned int arity = poseidon_legal_arities[arity_idx];
      unsigned int round_constants_len = arity * (poseidon_constants[arity].nof_upper_full_rounds + poseidon_constants[arity].nof_upper_full_rounds) + partial_rounds;
      unsigned int mds_matrix_len = arity * arity;      
      poseidon_constants[arity].rounds_constants = h_constants;
      poseidon_constants[arity].mds_matrix = poseidon_constants[arity].rounds_constants + round_constants_len;
      poseidon_constants[arity].pre_matrix = poseidon_constants[arity].mds_matrix + mds_matrix_len;
      poseidon_constants[arity].sparse_matrices = poseidon_constants[arity].pre_matrix + mds_matrix_len;
    }
    return eIcicleError::SUCCESS;
  }    

    // Size should be zero in order or equal to HashBackend::m_default_input_chunk_size.
    // Otherwise return eIcicleError::INVALID_ARGUMENT.
    eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const override    // DANNY - config isn't needed in cpu
    {
      ICICLE_LOG_DEBUG << "Poseidon CPU hash() " << size << " bytes, for type " << demangle<S>()
                       << ", batch=" << config.batch;

      if (size != 0 && size != m_default_input_chunk_size) {
        ICICLE_LOG_ERROR << "PoseidonBackendCPU:hash(...): size should be either zero or HashBackend::m_default_input_chunk_size";
        ICICLE_LOG_ERROR << "PoseidonBackendCPU:hash(...): size = " << size << ", m_default_input_chunk_size = " << m_default_input_chunk_size;
        return eIcicleError::INVALID_ARGUMENT;
      }
      print_input_bytes("Input", input, size);    // DEBUG
      
      unsigned int arity = m_default_input_chunk_size / sizeof(S);

      unsigned int alpha = poseidon_constants[arity].alpha;
      unsigned int nof_upper_full_rounds = poseidon_constants[arity].nof_upper_full_rounds;
      unsigned int nof_partial_rounds = poseidon_constants[arity].nof_partial_rounds;
      unsigned int nof_end_full_rounds = poseidon_constants[arity].nof_end_full_rounds;
      S*           rounds_constants = poseidon_constants[arity].rounds_constants;
      S*           mds_matrix = poseidon_constants[arity].mds_matrix;
      S*           pre_matrix = poseidon_constants[arity].pre_matrix;
      S*           sparse_matrices = poseidon_constants[arity].sparse_matrices;
      // Allocate temporary memory for intermediate calcs.
      S* tmp_fields = new S[arity];
      // Casting from limbs to scalar.
      const S* in_fields = (S*)(input);
      // Copy input scalar to the output (as a temp storage) to be used in the rounds.
      // *tmp_fields are used as a temp storage alog the calculations in this function.
      memcpy(tmp_fields, in_fields, arity * sizeof(S));

      // Add pre-round constants.
      for (int arity_idx=0; arity_idx<arity; arity_idx++) {
        tmp_fields[arity_idx] = tmp_fields[arity_idx] + *rounds_constants++;
      }

      // Upper full rounds.
      full_rounds(nof_upper_full_rounds, tmp_fields, rounds_constants);

      // Single full round with pre_matrix.
      for (int arity_idx=0; arity_idx<arity; arity_idx++) {
        // S box
        tmp_fields[arity_idx] = S::pow(tmp_fields[arity_idx], alpha);
        // Add round constants
        tmp_fields[arity_idx] = tmp_fields[arity_idx] + *rounds_constants++;
      }
      // Multiplication by matrix
      field_vec_sqr_full_matrix_mul(tmp_fields, pre_matrix, tmp_fields);

      // Partial rounds. Perform calculation only for the first element of *tmp_fields.
      for (int partial_rounds_idx=0; partial_rounds_idx<nof_partial_rounds; partial_rounds_idx++) {
        // S box
        tmp_fields[0] = S::pow(tmp_fields[0], alpha);
        // Add round constants
        tmp_fields[0] = tmp_fields[0] + *rounds_constants++;
        // Multiplication by sparse matrix.
        field_vec_sqr_sparse_matrix_mul(tmp_fields, &sparse_matrices[partial_rounds_idx*arity*arity], tmp_fields);
      }

      // Bottom full rounds.
      full_rounds(nof_end_full_rounds, tmp_fields, rounds_constants);

      // Last full round
      for (int arity_idx=0; arity_idx<arity; arity_idx++) {
        // S box
        tmp_fields[arity_idx] = S::pow(tmp_fields[arity_idx], alpha);
      }
      // Multiplication by MDS matrix
      field_vec_sqr_full_matrix_mul(tmp_fields, mds_matrix, tmp_fields);

      memcpy(output, (std::byte*)(&tmp_fields[1]), sizeof(S));

      delete[] tmp_fields;
      tmp_fields = nullptr;

      return eIcicleError::SUCCESS;
    }

    void field_vec_sqr_sparse_matrix_mul(const S* vec_in, const S* matrix_in, S* result) const {
      unsigned int arity = m_default_input_chunk_size / sizeof(S);
      S tmp_col_res[arity];     // Have to use temp storage because vec_in and result are the same storage.
      tmp_col_res[0] = S::from(0);
      for (int col_idx = 0; col_idx < arity; col_idx++) {   // Calc first column result.
        tmp_col_res[0] = tmp_col_res[0] + vec_in[col_idx] * matrix_in[col_idx * arity];
      }
      for (int col_idx = 1; col_idx < arity; col_idx++) {   // Calc rest columns results.
        tmp_col_res[col_idx] = S::from(0);
        tmp_col_res[col_idx] = vec_in[0] * matrix_in[col_idx] + vec_in[col_idx];
      }
      for (int col_idx = 0; col_idx < arity; col_idx++) {   // This copy is needed because vec_in and result storages are actually the same storage when calling to the function.
            result[col_idx] = tmp_col_res[col_idx];
      }
    }

    void field_vec_sqr_full_matrix_mul(const S* vec_in, const S* matrix_in, S* result) const {
      unsigned int arity = m_default_input_chunk_size / sizeof(S);
      S tmp_col_res[arity];     // Have to use temp storage because vec_in and result are the same storage.
      for (int col_idx = 0; col_idx < arity; col_idx++) {   // Columns of matrix.
        tmp_col_res[col_idx] = S::from(0);
        for (int row_idx = 0; row_idx < arity; row_idx++) {    // Matrix rows but also input vec columns.
          tmp_col_res[col_idx] = tmp_col_res[col_idx] + vec_in[row_idx] * matrix_in[row_idx * arity + col_idx];
        }
      }
      for (int col_idx = 0; col_idx < arity; col_idx++) {   // This copy is needed because vec_in and result storages are actually the same storage when calling to the function.
            result[col_idx] = tmp_col_res[col_idx];
      }
    }

    void full_rounds(const unsigned int nof_full_rounds, S* in_out_fields, const S* rounds_constants) const {
      unsigned int arity = m_default_input_chunk_size / sizeof(S);
      unsigned int alpha = poseidon_constants[arity].alpha;
      S*           mds_matrix = poseidon_constants[arity].mds_matrix;
      for (int full_rounds_idx=0; full_rounds_idx<nof_full_rounds-1; full_rounds_idx++) {   // Note that the number of full rounds is nof_full_rounds-1 because of the
                                                                                            // pre_matrix round (last round of upper full rounds) and the last round of
                                                                                            // the bottom fulld rounds that doesn't have sbox operation.
        for (int arity_idx=0; arity_idx<arity; arity_idx++) {
          // S box
          in_out_fields[arity_idx] = S::pow(in_out_fields[arity_idx], alpha);
          // Add round constants
          in_out_fields[arity_idx] = in_out_fields[arity_idx] + *rounds_constants++;
        }
        // Multiplication by matrix
        field_vec_sqr_full_matrix_mul(in_out_fields, mds_matrix, in_out_fields);
      }
    }
  };    // class PoseidonBackendCPU : public HashBackend

  static eIcicleError create_cpu_poseidon_hash_backend(
    const Device& device, unsigned arity, std::shared_ptr<HashBackend>& backend /*OUT*/, const scalar_t& phantom)
  {
    ICICLE_LOG_DEBUG << "in create_cpu_poseidon_hash_backend(arity=" << arity << ")";
    backend = std::make_shared<PoseidonBackendCPU<scalar_t>>(arity);
    return eIcicleError::SUCCESS;
  }

  REGISTER_CREATE_POSEIDON_BACKEND("CPU", create_cpu_poseidon_hash_backend);

} // namespace icicle