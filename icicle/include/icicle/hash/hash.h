#pragma once

#include <memory>
#include "icicle/hash/hash_config.h"
#include "icicle/backend/hash/hash_backend.h"

namespace icicle {

  /**
   * @brief Class representing a high-level hash interface.
   *
   * This class provides an interface for performing hash operations using a
   * device-specific backend hash implementation (e.g., Keccak, Blake2, Poseidon).
   * The actual hashing logic is delegated to the backend, allowing the user to
   * interact with a unified hash interface.
   */
  class Hash
  {
  public:
    /**
     * @brief Constructor for the Hash class.
     *
     * @param backend A shared pointer to a backend hash implementation.
     */
    Hash(std::shared_ptr<HashBackend> backend) : m_backend(std::move(backend)) {}

    /**
     * @brief Perform a hash operation.
     *
     * This function delegates the hash operation to the backend and returns the result.
     *
     * @param input Pointer to the input data as bytes.
     * @param size Number of bytes to hash. For batch case, this is a single chunk of data. Set 0 for default chunk
     * size.
     * @param config Configuration options for the hash operation.
     * @param output Pointer to the output data in bytes.
     * @return An error code of type eIcicleError indicating success or failure.
     */
    inline eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const
    {
      return m_backend->hash(input, size, config, output);
    }

    template <typename PREIMAGE, typename IMAGE>
    inline eIcicleError hash(const PREIMAGE* input, uint64_t size, const HashConfig& config, IMAGE* output) const
    {
      return hash((const std::byte*)input, size * sizeof(PREIMAGE), config, (std::byte*)output);
    }

    /**
     * @brief Get the default input chunk size.
     * @return The size of the default input chunk (optional).
     */
    inline uint64_t input_default_chunk_size() const { return m_backend->input_default_chunk_size(); }

    /**
     * @brief Get the output size in bytes.
     * @return The total number of output bytes.
     */
    inline uint64_t output_size() const { return m_backend->output_size(); }

  private:
    std::shared_ptr<HashBackend> m_backend; ///< Pointer to the backend that performs the actual hash operation.
  };

} // namespace icicle