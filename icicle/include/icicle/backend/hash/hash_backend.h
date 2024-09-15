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
     * @param default_input_chunk_size The default size of a single input chunk in bytes. Useful for Merkle trees.
     */
    HashBackend(uint64_t output_size, uint64_t default_input_chunk_size = 0)
        : m_output_size{output_size}, m_default_input_chunk_size{default_input_chunk_size}
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
     * @param size Number of bytes to hash. For batch case, this is a single chunk of data. Set 0 for default chunk
     * size.
     * @param config Hash configuration (e.g., async, device location).
     * @param output Pointer to the output data in bytes.
     * @return An error code of type eIcicleError indicating the result of the operation.
     */
    virtual eIcicleError
    hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const = 0;

    /**
     * @brief Get the default input chunk size.
     * @return The size of the input chunk in bytes.
     */
    inline uint64_t input_default_chunk_size() const { return m_default_input_chunk_size; }

    /**
     * @brief Get the output size in bytes for a single hash chunk.
     * @return The output size in bytes.
     */
    uint64_t output_size() const { return m_output_size; }

  protected:
    const uint64_t m_output_size;              ///< The number of output bytes produced by the hash.
    const uint64_t m_default_input_chunk_size; ///< Expected input chunk size for hashing operations.

    inline uint64_t get_single_chunk_size(uint64_t size_or_zero) const
    {
      auto size = (size_or_zero == 0) ? input_default_chunk_size() : size_or_zero;
      ICICLE_ASSERT(size > 0) << "Cannot infer hash size. Make sure to pass it to hasher.hash(...size...) or have "
                                 "default size for the hasher";
      return size;
    }
  };

} // namespace icicle