#pragma once

#include "icicle/common.h" // for limb
#include "icicle/hash/hash_config.h"

namespace icicle {

  /**
   * @brief Abstract base class for implementing device-specific hash functions.
   *
   * This class serves as the backend for the hashing operations and must be derived
   * by any concrete hash implementation (e.g., Keccak, Blake2, Poseidon). Each
   * derived class should implement the hashing logic for a specific device or environment.
   *
   * This class also holds information about the number of input/output limbs and
   * optional secondary input limbs that may be required during hash operations.
   */
  class HashBackend
  {
  public:
    /**
     * @brief Constructor for the HashBackend class.
     *
     * @param total_input Total number of input limbs.
     * @param total_output Total number of output limbs.
     * @param total_secondary Optional total number of secondary input limbs, default is 0.
     */
    HashBackend(uint64_t total_input, uint64_t total_output, uint64_t total_secondary = 0)
        : m_total_input_limbs(total_input), m_total_output_limbs(total_output),
          m_total_secondary_input_limbs(total_secondary)
    {
    }

    /**
     * @brief Virtual destructor to allow proper cleanup in derived classes.
     */
    virtual ~HashBackend() = default;

    /**
     * @brief Perform a single hash operation.
     *
     * This function should implement the logic for hashing a single input. Derived
     * classes must override this function to provide a concrete implementation of
     * the hash algorithm.
     *
     * @param input_limbs Pointer to the input data in limbs.
     * @param output_limbs Pointer to the output data in limbs.
     * @param config Hash configuration options (e.g., async, device location).
     * @param secondary_input_limbs Optional pointer to secondary input data for
     * hash algorithms that support additional inputs during intermediate stages.
     * @return An error code of type eIcicleError indicating the success or failure
     * of the operation.
     */
    virtual eIcicleError hash_single(
      const limb_t* input_limbs,
      limb_t* output_limbs,
      const HashConfig& config,
      const limb_t* secondary_input_limbs = nullptr) const = 0;

    /**
     * @brief Perform multiple hash operations in a batch.
     *
     * This function should implement the logic for hashing multiple inputs in a
     * batch. Derived classes must override this function to provide a concrete
     * implementation for batching the hash algorithm.
     *
     * @param input_limbs Pointer to the input data in limbs.
     * @param output_limbs Pointer to the output data in limbs.
     * @param nof_hashes Number of hashes to process.
     * @param config Hash configuration options (e.g., async, device location).
     * @param secondary_input_limbs Optional pointer to secondary input data.
     * @return An error code of type eIcicleError indicating the success or failure
     * of the operation.
     */
    virtual eIcicleError hash_many(
      const limb_t* input_limbs,
      limb_t* output_limbs,
      int nof_hashes,
      const HashConfig& config,
      const limb_t* secondary_input_limbs = nullptr) const = 0;

    /**
     * @brief Get the total number of input limbs.
     * @return The total number of input limbs as uint64_t.
     */
    uint64_t total_input_limbs() const { return m_total_input_limbs; }

    /**
     * @brief Get the total number of output limbs.
     * @return The total number of output limbs as uint64_t.
     */
    uint64_t total_output_limbs() const { return m_total_output_limbs; }

    /**
     * @brief Get the total number of secondary input limbs.
     * @return The total number of secondary input limbs as uint64_t.
     */
    uint64_t total_secondary_input_limbs() const { return m_total_secondary_input_limbs; }

  protected:
    const uint64_t m_total_input_limbs;           ///< Total number of input limbs.
    const uint64_t m_total_output_limbs;          ///< Total number of output limbs.
    const uint64_t m_total_secondary_input_limbs; ///< Total number of secondary input limbs (optional).
  };

} // namespace icicle