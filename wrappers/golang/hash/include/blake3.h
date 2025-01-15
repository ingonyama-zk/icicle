#include <stdint.h>
#include "hash.h"

#ifndef _BLAKE3_HASH
  #define _BLAKE3_HASH

  #ifdef __cplusplus
extern "C" {
  #endif

Hash* icicle_create_blake3(uint64_t default_input_chunk_size);

  #ifdef __cplusplus
}
  #endif

#endif
