#include <memory>
#include "icicle/hash/hash.h"
#include "icicle/errors.h"
#include "icicle/hash/keccak.h"
#include "icicle/hash/blake2s.h"

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
eIcicleError icicle_hasher_hash(
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
uint64_t icicle_hasher_output_size(HasherHandle hash_ptr)
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
eIcicleError icicle_hasher_delete(HasherHandle hash_ptr)
{
  if (!hash_ptr) return eIcicleError::INVALID_POINTER;
  delete hash_ptr;
  return eIcicleError::SUCCESS;
}

/*=================================================================================*
   The following are factories to create hasher objects of general-purpose hashes.
 *=================================================================================*/

/**
 * @brief Creates a Keccak-256 hash object.
 *
 * This function constructs a Hash object configured for Keccak-256.
 *
 * @param input_chunk_size Size of the input in bytes for the Keccak-256 hash.
 * @return HasherHandle A handle to the created Keccak-256 Hash object.
 */
HasherHandle icicle_create_keccak_256(uint64_t input_chunk_size)
{
  return new icicle::Hash(icicle::create_keccak_256_hash(input_chunk_size));
}

/**
 * @brief Creates a Keccak-512 hash object.
 *
 * This function constructs a Hash object configured for Keccak-512.
 *
 * @param input_chunk_size Size of the input in bytes for the Keccak-512 hash.
 * @return HasherHandle A handle to the created Keccak-512 Hash object.
 */
HasherHandle icicle_create_keccak_512(uint64_t input_chunk_size)
{
  return new icicle::Hash(icicle::create_keccak_512_hash(input_chunk_size));
}

/**
 * @brief Creates a SHA3-256 hash object.
 *
 * This function constructs a Hash object configured for SHA3-256.
 *
 * @param input_chunk_size Size of the input in bytes for the SHA3-256 hash.
 * @return HasherHandle A handle to the created SHA3-256 Hash object.
 */
HasherHandle icicle_create_sha3_256(uint64_t input_chunk_size)
{
  return new icicle::Hash(icicle::create_sha3_256_hash(input_chunk_size));
}

/**
 * @brief Creates a SHA3-512 hash object.
 *
 * This function constructs a Hash object configured for SHA3-512.
 *
 * @param input_chunk_size Size of the input in bytes for the SHA3-512 hash.
 * @return HasherHandle A handle to the created SHA3-512 Hash object.
 */
HasherHandle icicle_create_sha3_512(uint64_t input_chunk_size)
{
  return new icicle::Hash(icicle::create_sha3_512_hash(input_chunk_size));
}

/**
 * @brief Creates a Blake2s hash object.
 *
 * This function constructs a Hash object configured for Blake2s.
 *
 * @param input_chunk_size Size of the input in bytes for the Blake2s hash.
 * @return HasherHandle A handle to the created Blake2s Hash object.
 */
HasherHandle icicle_create_blake2s(uint64_t input_chunk_size)
{
  return new icicle::Hash(icicle::create_blake2s_hash(input_chunk_size));
}
}