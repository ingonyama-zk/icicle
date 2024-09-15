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
  };

  /**
   * @brief Initialize Poseidon constants with full configuration parameters.
   *
   * This function initializes a PoseidonConstants object with the provided configuration parameters.
   * The user provides arity, alpha, round constants, matrices, and other Poseidon-specific parameters.
   * The backend allocates and returns the initialized constants as a shared object.
   *
   * @param arity The arity (branching factor) of the Poseidon hash.
   * @param alpha Exponent used in the S-box function.
   * @param nof_partial_rounds Number of partial rounds.
   * @param nof_upper_full_rounds Number of full rounds at the beginning.
   * @param nof_end_full_rounds Number of full rounds at the end.
   * @param rounds_constants Array of round constants for Poseidon.
   * @param mds_matrix Array representing the MDS matrix.
   * @param pre_matrix Array representing the pre-processing matrix.
   * @param sparse_matrix Array representing the sparse matrix.
   * @param constants [OUT] Shared pointer to the initialized PoseidonConstants object.
   *
   * @return eIcicleError Error code indicating success or failure of the initialization.
   */
  template <typename S>
  eIcicleError poseidon_init_constants(
    unsigned arity,
    unsigned alpha,
    unsigned nof_partial_rounds,
    unsigned nof_upper_full_rounds,
    unsigned nof_end_full_rounds,
    const S* rounds_constants,
    const S* mds_matrix,
    const S* pre_matrix,
    const S* sparse_matrix,
    std::shared_ptr<PoseidonConstants<S>>& constants /*out*/);

  /**
   * @brief Initialize Poseidon constants with default values based on arity.
   *
   * This function initializes a PoseidonConstants object with default values, based only on the arity.
   * The backend will populate the constants object with pre-determined default values for Poseidon parameters.
   *
   * @param arity The arity (branching factor) of the Poseidon hash.
   * @param constants [OUT] Shared pointer to the initialized PoseidonConstants object.
   *
   * @return eIcicleError Error code indicating success or failure of the initialization.
   */
  template <typename S>
  eIcicleError
  poseidon_init_default_constants(unsigned arity, std::shared_ptr<PoseidonConstants<S>>& constants /*out*/);

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