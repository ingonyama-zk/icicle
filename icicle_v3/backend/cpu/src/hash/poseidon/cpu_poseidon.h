#include <cassert>
#include "icicle/fields/field.h"
#include "hash.h"                   // DEBUG - use common hash.h and not from the local dir.
#include "icicle/fields/field_config.h"

// typedef uint32_t limb_t;            // DEBUG. To remove after Hadar submit hash.h.

namespace icicle {
 
/**
 * @brief Poseidon hash constant struct.
 */
template <typename S>
struct PoseidonConstants {
  const S* m_rounds_constants;     ///< Rounds constants of this Poseidon hash.
  const S* m_mds_matrix;           ///< Full rounds MDS matrix.
  const S* m_pre_matrix;           ///< Pre-matrix (used at the last round of the upper full rounds).
  const S* m_sparse_round_matrix;  ///< Partial rounds MDS matrix.
  PoseidonConstants(const S* m_rounds_constants, const S* m_mds_matrix, const S* m_pre_matrix, const S* m_sparse_round_matrix) :
    m_rounds_constants(m_rounds_constants), m_mds_matrix(m_mds_matrix),
    m_pre_matrix(m_pre_matrix), m_sparse_round_matrix(m_sparse_round_matrix) {}
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
      const S*            rounds_constants,
      const S*            mds_matrix,
      const S*            pre_matrix,
      const S*            sparse_matrix
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
    const unsigned int    m_alpha;     ///< m_alpha of this Poseidon hash S-box.
    const unsigned int    m_nof_partial_rounds;     ///< Partial number of rounds of this Poseidon hash.
    const unsigned int    m_nof_upper_full_rounds;     ///< Number of full rounds at the beginning of this Poseidon hash.
    const unsigned int    m_nof_end_full_rounds;     ///< Number of full rounds at the end of this Poseidon hash.
    PoseidonConstants<S>  m_poseidon_constants;      ///< Structure that holds Poseidon hash round constants and MDS matrix values.

    void field_vec_sqr_full_matrix_mul(const S* vector, const S* matrix, S* result, const int arity) const;
    void field_vec_sqr_sparse_matrix_mul(const S* vector, const S* matrix, S* result, const int arity) const;
    void full_rounds(const unsigned int arity, const unsigned int alpha, const unsigned int nof_full_rounds, S* in_out_fields, const S*& rounds_constants, const S* mds_matrix) const;
};

template <typename S>
Poseidon<S>::Poseidon (
  const unsigned int  arity,
  const unsigned int  alpha,
  const unsigned int  nof_partial_rounds,
  const unsigned int  nof_upper_full_rounds,
  const unsigned int  nof_end_full_rounds,
  const S*            rounds_constants,
  const S*            mds_matrix,
  const S*            pre_matrix,
  const S*            sparse_matrix
  ) : Hash(arity*(S::TLC), arity*(S::TLC), 0), m_arity(arity), m_alpha(alpha), m_nof_partial_rounds(nof_partial_rounds),
      m_nof_upper_full_rounds(nof_upper_full_rounds), m_nof_end_full_rounds(nof_end_full_rounds),
      m_poseidon_constants{rounds_constants, mds_matrix, pre_matrix, sparse_matrix} {}

template <typename S>
eIcicleError Poseidon<S>::run_single_hash(const limb_t* input_limbs, limb_t* output_limbs, const HashConfig& config) const {
  // Allocate temporary memory for intermediate calcs.
  S* tmp_fields = new S[m_arity];
  // Casting from limbs to scalar.
  const S* in_fields = (S*)(input_limbs);
  // Copy input scalar to the output (as a temp storage) to be used in the rounds.
  // *tmp_fields are used as a temp storage alog the calculations in this function.
  for (int arity_idx=0; arity_idx<m_arity; arity_idx++) {
    tmp_fields[arity_idx] = in_fields[arity_idx];
  }

  // Add pre-round constants.
  const S* pre_round_constants = m_poseidon_constants.m_pre_matrix;    // Needed because m_poseidon_constants.m_pre_matrix couldn't be changed.
  const S* rounds_constants = m_poseidon_constants.m_rounds_constants;    // Needed because m_poseidon_constants.m_rounds_constants couldn't be changed.  
  for (int arity_idx=0; arity_idx<m_arity; arity_idx++) {
    tmp_fields[arity_idx] = tmp_fields[arity_idx] + *rounds_constants++;
  }

  // Upper full rounds.
  full_rounds(m_arity, m_alpha, m_nof_upper_full_rounds, tmp_fields, rounds_constants, m_poseidon_constants.m_mds_matrix);

  // Single full round with pre_matrix.
  for (int arity_idx=0; arity_idx<m_arity; arity_idx++) {
    // S box
    tmp_fields[arity_idx] = S::pow(tmp_fields[arity_idx], m_alpha);
    // Add round constants
    tmp_fields[arity_idx] = tmp_fields[arity_idx] + *rounds_constants++;
  }
  // Multiplication by matrix
  field_vec_sqr_full_matrix_mul(tmp_fields, m_poseidon_constants.m_pre_matrix, tmp_fields, m_arity);

  // Partial rounds. Perform calculation only for the first element of *tmp_fields.
  for (int partial_rounds_idx=0; partial_rounds_idx<m_nof_partial_rounds; partial_rounds_idx++) {
    // S box
    tmp_fields[0] = S::pow(tmp_fields[0], m_alpha);
    // Add round constants
    tmp_fields[0] = tmp_fields[0] + *rounds_constants++;
    // Multiplication by sparse matrix.
    field_vec_sqr_sparse_matrix_mul(tmp_fields, &m_poseidon_constants.m_sparse_round_matrix[partial_rounds_idx*m_arity*m_arity], tmp_fields, m_arity);
  }

  // Bottom full rounds.
  full_rounds(m_arity, m_alpha, m_nof_end_full_rounds, tmp_fields, rounds_constants, m_poseidon_constants.m_mds_matrix);

  // Last full round
  for (int arity_idx=0; arity_idx<m_arity; arity_idx++) {
    // S box
    tmp_fields[arity_idx] = S::pow(tmp_fields[arity_idx], m_alpha);
  }
  // Multiplication by MDS matrix
  field_vec_sqr_full_matrix_mul(tmp_fields, m_poseidon_constants.m_mds_matrix, tmp_fields, m_arity);

  memcpy(output_limbs, &tmp_fields[1].limbs_storage.limbs, S::TLC * sizeof(limb_t));

  delete[] tmp_fields;
  tmp_fields = nullptr;

  return eIcicleError::SUCCESS;
}     // eIcicleError Poseidon<S>::run_single_hash

template <typename S>
eIcicleError Poseidon<S>::run_multiple_hash(const limb_t *input_limbs, limb_t *output_limbs, int nof_hashes, const HashConfig& config, const limb_t *side_input_limbs) const {
  for (int hash_idx=0; hash_idx<nof_hashes; hash_idx++) {
    ICICLE_CHECK(run_single_hash(&input_limbs[hash_idx * m_arity * S::TLC], &output_limbs[hash_idx * S::TLC], config));
  }
  return eIcicleError::SUCCESS;
}

template <typename S>
void Poseidon<S>::field_vec_sqr_sparse_matrix_mul(const S* vec_in, const S* matrix_in, S* result, const int arity) const {
  S tmp_col_res[arity];     // Have to use temp storage because vec_in and result are the same storage.
  tmp_col_res[0] = S::from(0);
  for (int col_idx = 0; col_idx < arity; col_idx++) {   // Calc first column result.
    tmp_col_res[0] = tmp_col_res[0] + vec_in[col_idx] * matrix_in[col_idx * arity];
  }
  for (int col_idx = 1; col_idx < arity; col_idx++) {   // Calc rest columns results.
    tmp_col_res[col_idx] = S::from(0);
    tmp_col_res[col_idx] = vec_in[0] * matrix_in[col_idx] + vec_in[col_idx];
  }
  for (int col_idx = 0; col_idx < arity; col_idx++) {
        result[col_idx] = tmp_col_res[col_idx];
  }
}

template <typename S>
void Poseidon<S>::field_vec_sqr_full_matrix_mul(const S* vec_in, const S* matrix_in, S* result, const int arity) const {
  S tmp_col_res[arity];     // Have to use temp storage because vec_in and result are the same storage.
  for (int col_idx = 0; col_idx < arity; col_idx++) {   // Columns of matrix.
    tmp_col_res[col_idx] = S::from(0);
    for (int row_idx = 0; row_idx < arity; row_idx++) {    // Matrix rows but also input vec columns.
      tmp_col_res[col_idx] = tmp_col_res[col_idx] + vec_in[row_idx] * matrix_in[row_idx * arity + col_idx];
    }
  }
  for (int col_idx = 0; col_idx < arity; col_idx++) {
        result[col_idx] = tmp_col_res[col_idx];
  }
}

template <typename S>
void Poseidon<S>::full_rounds(const unsigned int arity, const unsigned int alpha, const unsigned int nof_full_rounds, S* in_out_fields, const S*& rounds_constants, const S* mds_matrix) const {
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
    field_vec_sqr_full_matrix_mul(in_out_fields, mds_matrix, in_out_fields, arity);
  }
}

}   // namespace icicle
