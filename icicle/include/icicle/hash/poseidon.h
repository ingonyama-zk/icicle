#pragma once

#include "icicle/hash/hash.h"

namespace icicle {

  /**
   * @brief Class representing the constants for Poseidon hash.
   *
   * This class will store the necessary constants for Poseidon hashing operations.
   * It will be managed and allocated by the backend and used when initializing the Poseidon hash.
   */
  template <typename S>
  class PoseidonConstants
  {
    // The backend will populate this class with Poseidon-specific constants.
    // This is intentionally left opaque and managed by the backend.

  public:
    PoseidonConstants() { ICICLE_LOG_DEBUG << "creating poseidon constants"; }
    virtual ~PoseidonConstants() { ICICLE_LOG_DEBUG << "deleting poseidon constants"; }
  };

  /**
   * @brief Initialize Poseidon constants with full configuration parameters.
   *
   * This function initializes a `PoseidonConstants` object with the provided configuration parameters.
   * The user must provide the arity (branching factor), alpha (exponent used in the S-box),
   * round constants, MDS matrix, non-sparse matrix, sparse matrices, and other Poseidon-specific parameters.
   * The function allocates and returns the initialized constants through the `constants` output parameter.
   *
   * @tparam S Type of the field elements used for constants (e.g., integers or field elements).
   *
   * @param arity The arity (branching factor) of the Poseidon hash (i.e., the number of inputs).
   * @param alpha Exponent used in the S-box function.
   * @param full_rounds_half The number of full rounds (half of the total rounds).
   * @param partial_rounds The number of partial rounds.
   * @param rounds_constants Pointer to the array of round constants for Poseidon.
   * @param mds_matrix Pointer to the array representing the MDS (Maximum Distance Separable) matrix.
   * @param non_sparse_matrix Pointer to the array representing the non-sparse matrix.
   * @param sparse_matrices Pointer to the array representing the sparse matrices.
   * @param domain_tag Pointer to the domain tag.
   * @param constants [OUT] Shared pointer to the initialized `PoseidonConstants` object.
   *
   * @return eIcicleError Error code indicating the success or failure of the initialization.
   */
  template <typename S>
  eIcicleError poseidon_init_constants(
    unsigned arity,
    unsigned alpha,
    unsigned full_rounds_half,
    unsigned partial_rounds,
    const S* rounds_constants,
    const S* mds_matrix,
    const S* non_sparse_matrix,
    const S* sparse_matrices,
    const S* domain_tag,
    std::shared_ptr<PoseidonConstants<S>>& constants /*out*/);

  /**
   * @brief Initialize Poseidon constants with default values based on arity.
   *
   * This function initializes a PoseidonConstants object with default values, for any supported arity.
   * The backend will populate the constants object with pre-determined default values for Poseidon parameters.
   *
   * @param constants [OUT] Shared pointer to the initialized PoseidonConstants object.
   *
   * @return eIcicleError Error code indicating success or failure of the initialization.
   */
  template <typename S>
  eIcicleError poseidon_init_default_constants(std::shared_ptr<PoseidonConstants<S>>& constants /*out*/);

  /**
   * @brief Create a Poseidon hash object using the shared PoseidonConstants.
   *
   * This function creates a Poseidon hash object, using the shared constants that were initialized by
   * the backend. The constants will be shared between the user and the PoseidonHasher.
   *
   * @param constants Shared pointer to the PoseidonConstants object, which holds hash-specific data.
   *
   * @return Hash Poseidon Hash object ready to perform hashing operations.
   */
  template <typename S>
  Hash create_poseidon_hash(std::shared_ptr<PoseidonConstants<S>> constants);
  struct Poseidon {
    template <typename S>
    inline static Hash create(std::shared_ptr<PoseidonConstants<S>> constants)
    {
      return create_poseidon_hash(constants);
    }
  };

} // namespace icicle