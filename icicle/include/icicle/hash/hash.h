#pragma once

#include <memory>
#include "icicle/hash/hash_config.h"
#include "icicle/backend/hash/hash_backend.h"

namespace icicle {

  /**
   * @brief Class representing a high-level hash interface.
   *
   * This class provides a unified interface for performing hash operations using a
   * device-specific backend hash implementation (e.g., Keccak, Blake2, Poseidon).
   * The actual hashing logic is delegated to the backend, allowing users to work with
   * the same interface regardless of the backend.
   */
  class Hash
  {
  public:
    /**
     * @brief Constructor for the Hash class.
     *
     * @param backend A shared pointer to a backend hash implementation.
     */
    explicit Hash(std::shared_ptr<HashBackend> backend) : m_backend(std::move(backend)) {}

    /**
     * @brief Perform a hash operation.
     *
     * Delegates the hash operation to the backend.
     *
     * @param input Pointer to the input data as bytes.
     * @param size Number of bytes to hash. If 0, the default chunk size is used.
     * @param config Configuration options for the hash operation.
     * @param output Pointer to the output data in bytes.
     * @return An error code of type eIcicleError indicating success or failure.
     */
    inline eIcicleError hash(const std::byte* input, uint64_t size, const HashConfig& config, std::byte* output) const
    {
      return m_backend->hash(input, size, config, output);
    }

    /**
     * @brief Perform a hash operation using typed data.
     *
     * Converts the input and output types to `std::byte` pointers and forwards the call to the backend.
     *
     * @tparam PREIMAGE The type of the input data.
     * @tparam IMAGE The type of the output data.
     * @param input Pointer to the input data.
     * @param size The number of elements of type `PREIMAGE` to hash.
     * @param config Configuration options for the hash operation.
     * @param output Pointer to the output data.
     * @return An error code of type eIcicleError indicating success or failure.
     */
    template <typename PREIMAGE, typename IMAGE>
    inline eIcicleError hash(const PREIMAGE* input, uint64_t size, const HashConfig& config, IMAGE* output) const
    {
      return hash(
        reinterpret_cast<const std::byte*>(input), size * sizeof(PREIMAGE), config,
        reinterpret_cast<std::byte*>(output));
    }

    /**
     * @brief Get the default input chunk size.
     * @return The size of the default input chunk in bytes.
     */
    inline uint64_t input_default_chunk_size() const { return m_backend->input_default_chunk_size(); }

    /**
     * @brief Get the output size in bytes.
     * @return The size of the output in bytes.
     */
    inline uint64_t output_size() const { return m_backend->output_size(); }

    const std::string& name() const { return m_backend->name(); }

  private:
    std::shared_ptr<HashBackend> m_backend; ///< Shared pointer to the backend performing the hash operation.
  };

} // namespace icicle