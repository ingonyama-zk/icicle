#include <stdbool.h>
#include <stdint.h>

#ifndef _HASH
  #define _HASH

  #ifdef __cplusplus
extern "C" {
  #endif

typedef struct Hash Hash;
typedef struct HashConfig HashConfig;

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
int icicle_hasher_hash(
  Hash* hash_ptr, const uint8_t* input_ptr, uint64_t input_len, const HashConfig* config, uint8_t* output_ptr);

/**
 * @brief Returns the output size in bytes for the hash operation.
 *
 * @param hash_ptr Handle to the Hash object.
 * @return uint64_t The size of the output in bytes.
 */
uint64_t icicle_hasher_output_size(Hash* hash_ptr);

/**
 * @brief Deletes the Hash object and cleans up its resources.
 *
 * @param hash_ptr Handle to the Hash object.
 * @return eIcicleError indicating success or failure.
 */
int icicle_hasher_delete(Hash* hash_ptr);

  #ifdef __cplusplus
}
  #endif

#endif