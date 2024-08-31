// #include "/home/administrator/users/danny/github/icicle/icicle_v3/include/icicle/fields/field.h"    // Miki: fix path - dont know how.
#include "icicle_v3/include/icicle/fields/field.h"    // Miki: fix path - dont know how.
#include "hash.h"
#include "icicle/fields/field_config.h"

typedef uint32_t limb_t;    // DEBUG. To remove after Hadar submit hash.h.

// using namespace field_config;
namespace icicle {

/**
 * @brief Poseidon hash constant struct.
 */
template <typename S>
struct PoseidonConstants {
  S* m_full_rounds_constants;   ///< Full rounds constants of this Poseidon hash.
                                    ///< Number of such constants is (m_nof_upper_full_rounds + m_nof_end_full_rounds) * m_arity.
  S* m_partial_rounds_constants;  ///< Partial rounds constants of this Poseidon hash.
                                ///< Number of such constants is m_nof_partial_rounds * m_arity.
  S* m_full_round_matrix;     ///< Full rounds MDS matrix.
  S* m_partial_round_matrix;  ///< Partial rounds MDS matrix.
  PoseidonConstants(S* m_full_rounds_constants, S* m_partial_rounds_constants, S* m_full_round_matrix, S* m_partial_round_matrix) : 
    m_full_rounds_constants(m_full_rounds_constants), m_partial_rounds_constants(m_partial_rounds_constants),
    m_full_round_matrix(m_full_round_matrix), m_partial_round_matrix(m_partial_round_matrix) {}
};

/**
 * @brief Poseidon hash class.
 */
// template <typename S = scalar_t>
template <typename S>
class Poseidon : public Hash {
  public:
    Poseidon (
      const unsigned int  arity,
      const unsigned int  alpha,
      const unsigned int  nof_partial_rounds,
      const unsigned int  nof_upper_full_rounds,
      const unsigned int  nof_end_full_rounds,
      S*                  full_rounds_constants,
      S*                  partial_rounds_constants,
      S*                  full_round_matrix,
      S*                  partial_round_matrix
    );

    /**
     * @brief Function to run a single Poseidon hash.
     * @param input_limbs Pointer to the Poseidon hash input limbs.
     * @param output_limbs Pointer to the Poseidon hash output limbs.
     * @return Error code of type eIcicleError.
     */
    eIcicleError run_single_hash (const limb_t *input_limbs, limb_t *output_limbs, const HashConfig& config) const;

    /**
     * @brief Function to run multiple Poseidon hashes.
     * @param input_limbs Pointer to the input limbs.
     * @param output_limbs Pointer to the output limbs.
     * @param nof_hashes Number of hashes to run.
     * @return Error code of type eIcicleError.
     */
    eIcicleError run_multiple_hash(const limb_t *input_limbs, limb_t *output_limbs, int nof_hashes, const HashConfig& config, const limb_t *side_input_limbs = nullptr) const;

  private:
    const unsigned int    m_arity;     ///< m_arity of this Poseidon hash.
    const unsigned int    m_alpha;     ///< m_alpha of this Poseidon hash S-box.debug
    const unsigned int    m_nof_partial_rounds;     ///< Partial number of rounds of this Poseidon hash.
    const unsigned int    m_nof_upper_full_rounds;     ///< Number of full rounds at the beginning of this Poseidon hash.
    const unsigned int    m_nof_end_full_rounds;     ///< Number of full rounds at the end of this Poseidon hash.
    PoseidonConstants<S>  m_poseidon_constants;      ///< Structure that holds Poseidon hash round constants and MDS matrix values.

    void field_vec_matrix_mul(const S* vector, const S* matrix, S* result, const int nof_rows, const int nof_columns) const;
    void full_round(const unsigned int arity, const unsigned int alpha, const unsigned int nof_full_rounds, S* in_out_fields, const S*& full_rounds_constants, const S* full_round_matrix);
};

template <typename S>
Poseidon<S>::Poseidon (
  const unsigned int  arity,
  const unsigned int  alpha,
  const unsigned int  nof_partial_rounds,
  const unsigned int  nof_upper_full_rounds,
  const unsigned int  nof_end_full_rounds,
  S*                  full_rounds_constants,
  S*                  partial_rounds_constants,
  S*                  full_round_matrix,
  S*                  partial_round_matrix
  ) : Hash(arity*(S::TLC), arity*(S::TLC), 0), m_arity(arity), m_alpha(alpha), m_nof_partial_rounds(nof_partial_rounds),
      m_nof_upper_full_rounds(nof_upper_full_rounds), m_nof_end_full_rounds(nof_end_full_rounds),
      m_poseidon_constants{full_rounds_constants, partial_rounds_constants, full_round_matrix, partial_round_matrix} {}

template <typename S>
eIcicleError Poseidon<S>::run_single_hash (const limb_t* input_limbs, limb_t* output_limbs, const HashConfig& config) const {
  // Casting from limbs to scalar.
  S* out_fields  = (S*)(output_limbs);
  S* in_field_op = (S*)(input_limbs);
  // Copy input scalar to the output (as a temp storage) to be used in the rounds.
  // *out_fields are used as a temp storage alog the calculations in this function.
  for (int arity_idx=0; arity_idx<m_arity; arity_idx++) {
    *out_fields++ = *in_field_op++;
  }
  // Upper full rounds.
  full_round(m_arity, m_alpha, m_nof_upper_full_rounds, out_fields, m_poseidon_constants.m_full_rounds_constants, m_poseidon_constants.m_full_round_matrix);

  // Partial rounds. Perform calculation only for the first element of *out_fields.
  for (int partial_rounds_idx=0; partial_rounds_idx<m_nof_partial_rounds; partial_rounds_idx++) {
    // Add round constants
    out_fields[0] = out_fields[0] + *m_poseidon_constants.m_partial_rounds_constants;
    m_poseidon_constants.m_partial_rounds_constants++;
    // S box
    out_fields[0] = S::pow(out_fields[0], m_alpha);
    // Multiplication by matrix
    // Miki: its a sparse matrix you don't need n^2 calculations - only n (will be done later).
    field_vec_matrix_mul(out_fields, m_poseidon_constants.m_partial_round_matrix, out_fields, m_arity, m_arity);
  }

  // Bottom full rounds.
  full_round(m_arity, m_alpha, m_nof_end_full_rounds, out_fields, m_poseidon_constants.m_full_rounds_constants, m_poseidon_constants.m_full_round_matrix);
  return eIcicleError::SUCCESS;
}

template <typename S>
eIcicleError Poseidon<S>::run_multiple_hash(const limb_t *input_limbs, limb_t *output_limbs, int nof_hashes, const HashConfig& config, const limb_t *side_input_limbs) const {
  for (int hash_idx=0; hash_idx<nof_hashes; hash_idx++) {
    ICICLE_CHECK(run_single_hash(input_limbs++, output_limbs++, config));
  }
  return eIcicleError::SUCCESS;
}

template <typename S>
void Poseidon<S>::field_vec_matrix_mul(const S* vec_in, const S* matrix_in, S* result, const int nof_rows, const int nof_columns) const {
  for (int col_idx = 0; col_idx < nof_columns; col_idx++) {
    result[col_idx] == S::from(0);
    for (int row_idx = 0; row_idx < nof_rows; row_idx++) {
      result[col_idx] = result[col_idx] + vec_in[col_idx] * matrix_in[col_idx * nof_columns + row_idx];
    }
  }
}

template <typename S>
void Poseidon<S>::full_round(const unsigned int arity, const unsigned int alpha, const unsigned int nof_full_rounds, S* in_out_fields, const S*& full_rounds_constants, const S* full_round_matrix) {
  for (int full_rounds_idx=0; full_rounds_idx<nof_full_rounds; full_rounds_idx++) {
    for (int arity_idx=0; arity_idx<arity; arity_idx++) {
      // Add round constants
      *in_out_fields = *in_out_fields++ + *full_rounds_constants++;
      // S box
      *in_out_fields = S::pow(*in_out_fields, alpha);
    }
    // Multiplication by matrix
    field_vec_matrix_mul(in_out_fields, full_round_matrix, in_out_fields, arity, arity);
  }
}

}   // namespace icicle
