#include <stdint.h>
#include "hash.h"

#ifndef _KECCAK_HASH
  #define _KECCAK_HASH

  #ifdef __cplusplus
extern "C" {
  #endif

Hash* icicle_create_keccak_256(uint64_t default_input_chunk_size);
Hash* icicle_create_keccak_512(uint64_t default_input_chunk_size);

  #ifdef __cplusplus
}
  #endif

#endif
