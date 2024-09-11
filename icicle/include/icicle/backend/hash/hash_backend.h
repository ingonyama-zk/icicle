#pragma once

#include "icicle/hash/hash_config.h"
#include <optional>
#include <cstddef>

namespace icicle {

  /**
   * @brief Abstract base class for device-specific hash function implementations.
   *
   * This class serves as the backend for hashing operations and must be derived
   * by any specific hash implementation (e.g., Keccak, Blake2, Poseidon). Each
   * derived class should implement the hashing logic for a particular device or environment.
   *
   */
  class HashBackend
  {
  public:
    /**
     * @brief Constructor for the HashBackend class.
     *
     * @param output_size The size of the output in bytes.
     * @param expected_input_chunk_size The size of a single input chunk. Useful for Merkle trees.
     */
    HashBackend(uint64_t output_size, uint64_t expected_input_chunk_size = 0)
        : m_output_size(output_size), m_expected_input_chunk_size(expected_input_chunk_size)
    {
    }

    /**
     * @brief Virtual destructor for cleanup in derived classes.
     */
    virtual ~HashBackend() = default;

    /**
     * @brief Perform a hash operation on a single chunk of input data.
     *
     * @param input Pointer to the input data in bytes.
     * @param config Hash configuration (e.g., async, device location).
     * @param output Pointer to the output data in bytes.
     * @return An error code of type eIcicleError indicating the result of the operation.
     */
    virtual eIcicleError hash(const std::byte* input, const HashConfig& config, std::byte* output) const = 0;

    /**
     * @brief Get the input chunk size, either from the HashConfig or from the expected default.
     *
     * This method retrieves the input chunk size from the provided HashConfig. If the configuration does not
     * specify a size, it falls back to the expected input chunk size for the hash.
     * The default is useful when building merkle trees, to store chunk-size inside the hasher.
     *
     * @param config The HashConfig object containing the configuration details for the hash operation.
     * @return The size of the input chunk in bytes.
     * @throws An assertion failure if the chunk size is invalid (zero).
     */
    inline uint64_t input_chunk_size() const { return m_expected_input_chunk_size; }
    inline uint64_t input_chunk_size(const HashConfig& config) const
    {
      // Default to the value from the config if present, otherwise use the expected chunk size
      const uint64_t chunk_size = (config.input_chunk_size > 0) ? config.input_chunk_size : input_chunk_size();

      // Ensure the chunk size is valid
      ICICLE_ASSERT(chunk_size > 0) << "Input chunk size for hash is unknown. Ensure it is set in the HashConfig or "
                                       "provided when constructing the hasher.";

      return chunk_size;
    }

    /**
     * @brief Get the output size in bytes for a single hash chunk.
     * @return The output size in bytes.
     */
    uint64_t output_size() const { return m_output_size; }

  protected:
    const uint64_t m_output_size;               ///< The number of output bytes produced by the hash.
    const uint64_t m_expected_input_chunk_size; ///< Expected input chunk size for hashing operations.
  };

} // namespace icicle