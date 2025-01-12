#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "blake3_impl.h"

void blake3_compress_in_place(
  uint32_t cv[8], const uint8_t block[BLAKE3_BLOCK_LEN], uint8_t block_len, uint64_t counter, uint8_t flags)
{
  blake3_compress_in_place_portable(cv, block, block_len, counter, flags);
}

void blake3_compress_xof(
  const uint32_t cv[8],
  const uint8_t block[BLAKE3_BLOCK_LEN],
  uint8_t block_len,
  uint64_t counter,
  uint8_t flags,
  uint8_t out[64])
{
  blake3_compress_xof_portable(cv, block, block_len, counter, flags, out);
}

void blake3_xof_many(
  const uint32_t cv[8],
  const uint8_t block[BLAKE3_BLOCK_LEN],
  uint8_t block_len,
  uint64_t counter,
  uint8_t flags,
  uint8_t out[64],
  size_t outblocks)
{
  if (outblocks == 0) {
    // The current assembly implementation always outputs at least 1 block.
    return;
  }

  for (size_t i = 0; i < outblocks; ++i) {
    blake3_compress_xof(cv, block, block_len, counter + i, flags, out + 64 * i);
  }
}

void blake3_hash_many(
  const uint8_t* const* inputs,
  size_t num_inputs,
  size_t blocks,
  const uint32_t key[8],
  uint64_t counter,
  bool increment_counter,
  uint8_t flags,
  uint8_t flags_start,
  uint8_t flags_end,
  uint8_t* out)
{
  blake3_hash_many_portable(
    inputs, num_inputs, blocks, key, counter, increment_counter, flags, flags_start, flags_end, out);
}

// The dynamically detected SIMD degree of the current platform.
size_t blake3_simd_degree(void) { return 1; }
