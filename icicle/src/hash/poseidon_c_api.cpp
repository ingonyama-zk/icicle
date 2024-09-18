#include <memory>
#include "icicle/hash/hash.h"
#include "icicle/hash/poseidon.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"

using namespace field_config;
using namespace icicle;

extern "C" {
// Define opaque types to be used in the C API
typedef std::shared_ptr<icicle::PoseidonConstants<scalar_t>>* PoseidonConstantsHandle;
typedef icicle::Hash* HasherHandle;

/**
 * @brief Initialize Poseidon constants with full configuration parameters.
 *
 * This function initializes Poseidon constants and returns a handle to the initialized constants.
 *
 * @param arity The arity (branching factor) of the Poseidon hash.
 * @param alpha Exponent used in the S-box function.
 * @param full_rounds_half The number of full rounds (half of the total rounds).
 * @param partial_rounds The number of partial rounds.
 * @param rounds_constants Pointer to the array of round constants for Poseidon.
 * @param mds_matrix Pointer to the array representing the MDS matrix.
 * @param non_sparse_matrix Pointer to the array representing the non-sparse matrix.
 * @param sparse_matrices Pointer to the array representing the sparse matrices.
 * @param domain_tag Pointer to the domain tag.
 * @param constants [OUT] Handle to the initialized PoseidonConstants object.
 *
 * @return eIcicleError indicating success or failure of the initialization.
 */
eIcicleError CONCAT_EXPAND(FIELD, poseidon_init_constants)(
  unsigned arity,
  unsigned alpha,
  unsigned full_rounds_half,
  unsigned partial_rounds,
  const uint64_t* rounds_constants,
  const uint64_t* mds_matrix,
  const uint64_t* non_sparse_matrix,
  const uint64_t* sparse_matrices,
  const uint64_t* domain_tag,
  PoseidonConstantsHandle* constants /*output*/);

/**
 * @brief Initialize Poseidon constants with default values based on arity.
 *
 * This function initializes Poseidon constants with default values and returns a handle to the constants.
 *
 * @param arity The arity (branching factor) of the Poseidon hash.
 * @param constants [OUT] Handle to the initialized PoseidonConstants object.
 *
 * @return eIcicleError indicating success or failure of the initialization.
 */
eIcicleError CONCAT_EXPAND(FIELD, poseidon_init_default_constants)(unsigned arity, PoseidonConstantsHandle constants);

/**
 * @brief Delete the PoseidonConstants object.
 *
 * Cleans up and deallocates the PoseidonConstants object.
 *
 * @param constants Handle to the PoseidonConstants object.
 *
 * @return eIcicleError indicating success or failure of the deletion.
 */
eIcicleError CONCAT_EXPAND(FIELD, poseidon_delete_constants)(PoseidonConstantsHandle constants);

/**
 * @brief Creates a Poseidon hash object.
 *
 * This function constructs a Hash object configured for Poseidon with the provided constants.
 *
 * @param constants Handle to the initialized PoseidonConstants object.
 * @return HasherHandle A handle to the created Poseidon Hash object.
 */
HasherHandle CONCAT_EXPAND(FIELD, create_poseidon_hasher)(PoseidonConstantsHandle constants)
{
  return new icicle::Hash(icicle::create_poseidon_hash<scalar_t>(*constants));
}
}