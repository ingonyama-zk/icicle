#pragma once

#include <memory>
#include "icicle/common.h" // for limb
#include "icicle/hash/hash_config.h"
#include "icicle/backend/hash/hash_backend.h"

namespace icicle {

  /**
   * @brief Class representing a hash object.
   *
   * This class encapsulates a backend hash implementation (e.g., Keccak, Blake2, Poseidon)
   * and provides a high-level interface for performing hash operations. The user
   * interacts with this class, while the actual hashing logic is delegated to the backend.
   */
  class Hash
  {
  public:
    /**
     * @brief Constructor for the Hash class.
     *
     * @param backend Shared pointer to the backend hash implementation.
     */
    Hash(std::shared_ptr<HashBackend> backend) : m_backend(std::move(backend)) {}

    /**
     * @brief Perform a single hash operation.
     *
     * This function delegates the hashing operation to the backend and returns the result.
     *
     * @param input_limbs Pointer to the input data in limbs.
     * @param output_limbs Pointer to the output data in limbs.
     * @param config Configuration options for the hash operation.     
     * @return An error code of type eIcicleError indicating success or failure.
     */
    eIcicleError hash_single(
      const limb_t* input_limbs,
      limb_t* output_limbs,
      const HashConfig& config) const
    {
      return m_backend->hash_single(input_limbs, output_limbs, config);
    }

    /**
     * @brief Perform multiple hash operations in a batch.
     *
     * This function delegates the batching of hashes to the backend and returns the result.
     *
     * @param input_limbs Pointer to the input data in limbs.
     * @param output_limbs Pointer to the output data in limbs.
     * @param nof_hashes Number of hashes to perform.
     * @param config Configuration options for the hash operation.     
     * @return An error code of type eIcicleError indicating success or failure.
     */
    eIcicleError hash_many(
      const limb_t* input_limbs,
      limb_t* output_limbs,
      int nof_hashes,
      const HashConfig& config) const
    {
      return m_backend->hash_many(input_limbs, output_limbs, nof_hashes, config);
    }

    /**
     * @brief Get the total number of input limbs.
     * @return The total number of input limbs as uint64_t.
     */
    uint64_t total_input_limbs() const { return m_backend->total_input_limbs(); }

    /**
     * @brief Get the total number of output limbs.
     * @return The total number of output limbs as uint64_t.
     */
    uint64_t total_output_limbs() const { return m_backend->total_output_limbs(); }    

  private:
    std::shared_ptr<HashBackend> m_backend; ///< Pointer to the backend that performs the actual hash operation.
  };

} // namespace icicle