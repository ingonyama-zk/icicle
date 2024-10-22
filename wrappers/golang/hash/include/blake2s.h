#include <stdint.h>
#include "hash.h"

#ifndef _BLAKE2S_HASH
  #define _BLAKE2S_HASH

  #ifdef __cplusplus
extern "C" {
  #endif

Hash* icicle_create_blake2s(uint64_t default_input_chunk_size);

  #ifdef __cplusplus
}
  #endif

#endif
