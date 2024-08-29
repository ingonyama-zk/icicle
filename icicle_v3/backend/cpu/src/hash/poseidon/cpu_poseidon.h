// #include "icicle_v3/include/icicle/fields/field.h"
#include "/home/administrator/users/danny/github/icicle/icicle_v3/include/icicle/fields/field.h" // Miki: fix path
#include "hash.h"
#include "icicle/fields/field_config.h"

typedef uint32_t limb_t;    // DEBUG. To remove after Hadar submit hash.h.

// using namespace field_config;
namespace icicle {

/**
 * @brief Poseidon hash class.
 */
// template <typename S = scalar_t>
template <typename S>
class Poseidon : public Hash {
  private:// Miki: remove private down - public should appear first
    const unsigned int arity;     ///< Arity of this Poseidon hash.
    const unsigned int alpha;     ///< Alpha of this Poseidon hash S-box.
    const unsigned int nof_total_rounds;     ///< Total number of rounds (full + partial) of this Poseidon hash.
    const unsigned int nof_full_rounds_half;     ///< Number of partial rounds of this Poseidon hash.
    const S* full_rounds_constants;   ///< Full rounds constants of this Poseidon hash.
                                      ///< Number of such constants is 2 * nof_full_rounds_half * arity.
    const S* partial_rounds_constants;  ///< Partial rounds constants of this Poseidon hash.
                                  ///< Number of such constants is nof_total_rounds - 2 * nof_full_rounds_half.
    const S* full_round_matrix;     ///< Full rounds MDS matrix.
    const S* partial_round_matrix;  ///< Partial rounds MDS matrix.
// Miki: add to member variables m_
  public:
    Poseidon (
      const unsigned int arity,
      const unsigned int alpha,
      const unsigned int nof_total_rounds,    // Miki: don't you want nof_partial_rounds
      const unsigned int nof_full_rounds_half,  // Miki: is it half or full?
      // Miki: do we want a different number of full rounds for the begining and the end? check with Kartik

      // Miki: wrap all the constants in a single struct and we'll provide them part of icicle
      const S* full_rounds_constants,
      const S* partial_rounds_constants,
      const S* full_round_matrix,
      const S* partial_round_matrix
    );

    /**
     * @brief Function to run a single Poseidon hash.
     * @param input_limbs Pointer to the Poseidon hash input limbs.
     * @param output_limbs Pointer to the Poseidon hash output limbs.
     * @return Error code of type eIcicleError.
     */
    // Miki: needs to be virtual
    eIcicleError run_single_hash (const limb_t *input_limbs, limb_t *output_limbs, const HashConfig& config) const;

    /**
     * @brief Function to run multiple Poseidon hashes.
     * @param input_limbs Pointer to the input limbs.
     * @param output_limbs Pointer to the output limbs.
     * @param nof_hashes Number of hashes to run.
     * @return Error code of type eIcicleError.
     */
    // Miki: needs to be virtual
    eIcicleError run_multiple_hash(const limb_t *input_limbs, limb_t *output_limbs, int nof_hashes, const HashConfig& config, const limb_t *side_input_limbs = nullptr) const;

    // Miki: should be private
    void field_vec_matrix_mul(const S* vector, const S* matrix, S* result, const int nof_rows, const int num_columns) const;

};

template <typename S>
Poseidon<S>::Poseidon (
  const unsigned int arity,
  const unsigned int alpha,
  const unsigned int nof_total_rounds,
  const unsigned int nof_full_rounds_half,
  const S* full_rounds_constants,
  const S* partial_rounds_constants,
  const S* full_round_matrix,
  const S* partial_round_matrix
  // Miki: (0,0,0,) is wrong
  ) : Hash(0, 0, 0), arity(arity), alpha(alpha), nof_total_rounds(nof_total_rounds),
      nof_full_rounds_half(nof_full_rounds_half), full_rounds_constants(full_rounds_constants),
      partial_rounds_constants(partial_rounds_constants), full_round_matrix(full_round_matrix),
      partial_round_matrix(partial_round_matrix) {}

template <typename S>
eIcicleError Poseidon<S>::run_single_hash (const limb_t* input_limbs, limb_t* output_limbs, const HashConfig& config) const {
  // Casting from limbs to scalar.
  S* out_fields  = (S*)(output_limbs);
  S* in_field_op = (S*)(input_limbs);
  const S* tmp_full_rounds_constants = full_rounds_constants; // Miki: why is that
  const S* tmp_partial_rounds_constants = partial_rounds_constants; // Miki: why is that
  // Copy input scalar to the output (as a temp storage) to be used in the rounds.
  // *out_fields are used as a temp storage alog the calculations in this function.
  for (int arity_idx=0; arity_idx<arity; arity_idx++) {    // Miki: there is a perliminary calculation that is missing - you can use this calc to send result to output and avoid this for loop
    *out_fields++ = *in_field_op++;
  }
  // Upper full rounds.
  for (int full_rounds_half_idx=0; full_rounds_half_idx<nof_full_rounds_half; full_rounds_half_idx++) {
    for (int arity_idx=0; arity_idx<arity; arity_idx++) {
      // S box
      // Miki: use array syntaxt out_fields[ariti_idx]
      *out_fields = S::pow(*out_fields, alpha);
      // Add round constants
      *out_fields = *out_fields++ + *tmp_full_rounds_constants++;
    }
    // Miki: you didnt reset out_field so its out of memory range
    // Multiplication by matrix
    field_vec_matrix_mul(out_fields, full_round_matrix, out_fields, arity, arity);
  }
  // Partial rounds. Perform calculation only for the first element of *out_fields.
  for (int partial_rounds_idx=0; partial_rounds_idx<(nof_total_rounds-2*nof_full_rounds_half); partial_rounds_idx++) {
    // S box
    *out_fields = S::pow(*out_fields, alpha);
    // Add round constants
    *out_fields = *out_fields++ + *tmp_partial_rounds_constants++;
    // Multiplication by matrix
    // Miki: its a sparse matrix you don't need n^2 calculations - only n
    field_vec_matrix_mul(out_fields, partial_round_matrix, out_fields, arity, arity);
  }

  // Miki: use a function for partial round and call it twice  
  // Bottom full rounds.
  for (int full_rounds_half_idx=0; full_rounds_half_idx<nof_full_rounds_half; full_rounds_half_idx++) {
    for (int arity_idx=0; arity_idx<arity; arity_idx++) {
      // S box
      *out_fields = S::pow(*out_fields, alpha);
      // Add round constants
      *out_fields = *out_fields++ + *tmp_full_rounds_constants++;
    }
    // Multiplication by matrix
    field_vec_matrix_mul(out_fields, full_round_matrix, out_fields, arity, arity);
  }
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
void Poseidon<S>::field_vec_matrix_mul(const S* vec_in, const S* matrix_in, S* result, const int nof_rows, const int num_columns) const {
  for (int i = 0; i < num_columns; i++) {
    *result == S::from(0);
    for (int j = 0; j < nof_rows; j++) {
      // Miki: use array syntaxt 
      *result = *result + *vec_in * *matrix_in;
      vec_in++;
      matrix_in++;
    }
    result++;
  }
}

}   // namespace icicle
