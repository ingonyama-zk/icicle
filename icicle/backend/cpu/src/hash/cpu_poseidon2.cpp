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
      }                                              // switch (T) {
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
      int size_in_scalars = size / sizeof(S);
      if (size_in_scalars > (m_use_domain_tag ? m_t - 1 : m_t)) {    // Sponge function. Check input size granularity.
        is_sponge = true;
        // Capacity width (in scalars) = 1.
        // Output width (in scalars) = 1.
        if ((m_use_domain_tag ? size_in_scalars : size_in_scalars - 1) % (m_t - 1) != 0) {
          ICICLE_LOG_ERROR
            << "Padding isn't supported for sponge function hash. The following should be true: ((m_use_domain_tag ? size : size-1) % (m_t-1) != 0).\n";
          return eIcicleError::INVALID_ARGUMENT;
        }
        if (config.batch != 1) {
          ICICLE_LOG_ERROR
            << "The only suppoorted value of config.batch is 1.\n";
          return eIcicleError::INVALID_ARGUMENT;
        }
      } 
      else if (size_in_scalars < (m_use_domain_tag ? m_t - 1 : m_t)) {
        ICICLE_LOG_ERROR
          << "Padding isn't supported for sponge function hash. The following should be true: ((use_domain_tag ? size + 1 : size) % T = 0).\n";
        return eIcicleError::INVALID_ARGUMENT;
      }

      const unsigned int T = m_t;
      bool is_unsupported_T_for_this_field = poseidon2_constants[T].nof_upper_full_rounds == 0;
      if (is_unsupported_T_for_this_field) {
        ICICLE_LOG_ERROR << "Unsupported poseidon width (t=" << T << ") for this field! Planned for next version";
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
      int sponge_nof_hashers = m_use_domain_tag ? (size_in_scalars / arity) :  ((size_in_scalars - 1) / (arity - 1));
      int tmp_fields_nof_scalars = is_sponge ? (T * sponge_nof_hashers) : (T * config.batch);
      // std::cout << "tmp_fields_nof_scalars = " << tmp_fields_nof_scalars << std::endl;
      S* tmp_fields = new S[tmp_fields_nof_scalars];
      std::cout << "tmp_fields (new) = " << tmp_fields << std::endl;
      const S* in_fields = (S*)(input);
      if (m_use_domain_tag) {      
        if (is_sponge) {
          // std::cout << "is_sponge sponge_nof_hashers = " << sponge_nof_hashers << std::endl;
          // Domain tag exists only for the first hasher. For the rest of the hashers this
          // input is undefined at this stage and its value will be set later.
          // tmp_fields = {{dt, in0}, {undef, in1}, {undef, in2}, etc.}
          memcpy(tmp_fields, &m_domain_tag, sizeof(S));   
          for (int hasher_idx = 0; hasher_idx < sponge_nof_hashers; hasher_idx++) {
            memcpy(tmp_fields + 1, in_fields, (T - 1) * sizeof(S));
            in_fields += (T - 1);
            tmp_fields += T;
          }
          tmp_fields -= T * sponge_nof_hashers;
        }
        else {    // Not a sponge function. Input of each hash should have domain tag at its input.
          // tmp_fields = {{dt, in0}, {dt, in1}, {dt, in2}, etc.}
          for (int batch_idx = 0; batch_idx < config.batch; batch_idx++) {
            memcpy(tmp_fields, &m_domain_tag, sizeof(S));
            memcpy(tmp_fields + 1, in_fields, (T - 1) * sizeof(S));
            in_fields += (T - 1);
            tmp_fields += T;
          }
          tmp_fields -= T * config.batch;
        }
      }
      else {    // There is no domain tag.
        if (is_sponge) {
          // tmp_fields = {{in0 (T inputs)}, {undef, in1}, {under, in2}, etc.}
          // std::cout << "is_sponge sponge_nof_hashers = " << sponge_nof_hashers << std::endl;
          memcpy(tmp_fields, in_fields, T * sizeof(S)); // 1st hasher uses T inputs.
          // print_state("tmp_fields 1st hasher external input: ", tmp_fields);
          in_fields += T;
          tmp_fields += T;
          // Rest of the hashers use T-1 inputs and 1 output from the previous hasher.
          // The value of the 1st input of each of these hashers is undef at this stage.
          for (int hasher_idx = 1; hasher_idx < sponge_nof_hashers; hasher_idx++) {
            memcpy(&tmp_fields[1], in_fields, (T - 1) * sizeof(S));
            // print_state("tmp_fields 2nd hasher external input (1st scalar doesnt matter): ", tmp_fields);
            in_fields += (T - 1);
            tmp_fields += T;
          }
          tmp_fields -= T * sponge_nof_hashers;
        }
        else {
          // tmp_fields = {{in0 (T inputs)}, {in1 (T inputs)}, {in2 (T inputs)}, etc.}
          memcpy(tmp_fields, in_fields, T * config.batch * sizeof(S));
        }
      }

      // std::cout << "tmp_fields (start processing) = " << tmp_fields << std::endl;
      if (is_sponge) {
        S* tmp_fields_tmp_ptr;    // This pointer is used to assist in addition of the hasher outputs
                                  // with new inputs.
        // Call hash_single for hasher[0]
        // print_state("tmp_fields input to the first hasher: ", tmp_fields);        
        eIcicleError err = hash_single(tmp_fields /* input */, tmp_fields /* output */,
          alpha, nof_upper_full_rounds, nof_partial_rounds, nof_bottom_full_rounds,
          rounds_constants, mds_matrix, partial_matrix_diagonal_m1);
        // print_state("tmp_fields output of the first hasher: ", tmp_fields);          
        tmp_fields[T] = tmp_fields[0];    // Current first output is an input to the next hasher.
        tmp_fields_tmp_ptr = tmp_fields;  // Save current pointer.
        tmp_fields += T;
        for (int hasher_idx = 1; hasher_idx < sponge_nof_hashers; hasher_idx++) {
          // std::cout << "hasher_idx = " << hasher_idx << std::endl;
          // print_state("tmp_fields taken from external input (1st scalar doesnt matter): ", tmp_fields);
          // print_state("tmp_fields taken from prev hasher: ", tmp_fields_tmp_ptr);          
          // The first output of the prev hasher is the first input of the current hasher.
          // The T-1 new inputs of the current hasher should be added to the T-1 outputs of the
          // prev hasher (starting fom index 1).
          for (int i = 1; i < T; i++) {
            tmp_fields[i] = tmp_fields_tmp_ptr[i] + tmp_fields[i];
          }
          // print_state("tmp_fields after addition: ", tmp_fields);          
          tmp_fields_tmp_ptr = tmp_fields;    // Save current pointer.
          eIcicleError err = hash_single(tmp_fields /* input */, tmp_fields /* output */,
            alpha, nof_upper_full_rounds, nof_partial_rounds, nof_bottom_full_rounds,
            rounds_constants, mds_matrix, partial_matrix_diagonal_m1);
          // print_state("tmp_fields output of the non-first hasher: ", tmp_fields);
          if (err != eIcicleError::SUCCESS) return err;
          if (hasher_idx != sponge_nof_hashers - 1)   // Not to do in the last loop to prevent mem leak.
            tmp_fields[T] = tmp_fields[0];    // Current first output is an input to the next hasher.
          tmp_fields += T;  // Proceed to the next hash.
        }
        tmp_fields -= T;    // Rollback to the last hasher output.
        memcpy(output, (std::byte*)(&tmp_fields[1]), sizeof(S));
        tmp_fields -= T * (sponge_nof_hashers - 1);
      }
      else {    // Not a sponge function.
        for (int batch_hash_idx = 0; batch_hash_idx < config.batch; batch_hash_idx++) {
          // print_state("tmp_fields before hash_single: ", tmp_fields);
          eIcicleError err = hash_single(tmp_fields /* input */, tmp_fields /* output */,
            alpha, nof_upper_full_rounds, nof_partial_rounds, nof_bottom_full_rounds,
            rounds_constants, mds_matrix, partial_matrix_diagonal_m1);
          // print_state("tmp_fields after hash_single: ", tmp_fields);
          if (err != eIcicleError::SUCCESS) return err;
          memcpy(output, (std::byte*)(&tmp_fields[1]), sizeof(S));
          tmp_fields += T;
          output += sizeof(S);
        }
        tmp_fields -= T * config.batch;
      }

      // std::cout << "tmp_fields (delete) = " << tmp_fields<< std::endl;
      delete[] tmp_fields;
      tmp_fields = nullptr;

      return eIcicleError::SUCCESS;
    }

  private:
    // DEBUG start. Do not remove!!!
    void print_state(std::string str, S* state_to_print)  const {
      std::cout << str << std::endl;
      unsigned int T = m_t;
      for (int state_idx = 0; state_idx < T; state_idx++) { // Columns of matrix.
        std::cout << std::hex << state_to_print[state_idx] << std::endl;
      }
    }
    void print_matrix(std::string str, S* matrix_to_print)  const {
      std::cout << str << std::endl;
      unsigned int T = m_t;
      for (int matrix_idx = 0; matrix_idx < T*T; matrix_idx++) { // Columns of matrix.
        std::cout << std::hex << matrix_to_print[matrix_idx] << std::endl;
      }
    }
    // DEBUG end

    // This function performs a single hash according to parameters in the poseidon2_constants[] struct.
    // eIcicleError hash_single(const std::byte* input, std::byte* output) const
    eIcicleError hash_single(S* tmp_fields, S* hasher_output, int alpha, int nof_upper_full_rounds,
      int nof_partial_rounds, int nof_bottom_full_rounds,
      S* rounds_constants, S* mds_matrix, S* partial_matrix_diagonal_m1) const
    {
      const unsigned int T = m_t;

      // Pre-rounds full matrix multiplication.
      full_matrix_mul_by_vector(tmp_fields, mds_matrix, tmp_fields);
      // print_matrix("mds_matrix: ", mds_matrix);
      // print_state("Pre-rounds mds matrix multiplication: ", tmp_fields);

      // Upper full rounds.
      full_rounds(nof_upper_full_rounds, tmp_fields, rounds_constants);
      // print_state("Upper full rounds: ", tmp_fields);

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

      // delete[] tmp_fields;
      // tmp_fields = nullptr;

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