#include <memory>
#include "icicle/hash/hash.h"
#include "icicle/errors.h"

extern "C" {
// Define a type for the HasherHandle (which is a pointer to Hash)
typedef icicle::Hash* HasherHandle;

/**
 * @brief Hashes input data and stores the result in the output buffer.
 *
 * @param hash_ptr Handle to the Hash object.
 * @param input_ptr Pointer to the input data.
 * @param input_len Length of the input data in bytes.
 * @param config Pointer to the HashConfig object.
 * @param output_ptr Pointer to the output buffer to store the result.
 * @return eIcicleError indicating success or failure.
 */
eIcicleError hasher_hash(
  HasherHandle hash_ptr,
  const uint8_t* input_ptr,
  uint64_t input_len,
  const icicle::HashConfig* config,
  uint8_t* output_ptr)
{
  if (!hash_ptr) return eIcicleError::INVALID_POINTER;
  return hash_ptr->hash(
    reinterpret_cast<const std::byte*>(input_ptr), input_len, *config, reinterpret_cast<std::byte*>(output_ptr));
}

/**
 * @brief Returns the output size in bytes for the hash operation.
 *
 * @param hash_ptr Handle to the Hash object.
 * @return uint64_t The size of the output in bytes.
 */
uint64_t hasher_output_size(HasherHandle hash_ptr)
{
  if (!hash_ptr) return 0;
  return hash_ptr->output_size();
}

/**
 * @brief Deletes the Hash object and cleans up its resources.
 *
 * @param hash_ptr Handle to the Hash object.
 * @return eIcicleError indicating success or failure.
 */
eIcicleError hasher_delete(HasherHandle hash_ptr)
{
  if (!hash_ptr) return eIcicleError::INVALID_POINTER;
  delete hash_ptr;
  return eIcicleError::SUCCESS;
}
}