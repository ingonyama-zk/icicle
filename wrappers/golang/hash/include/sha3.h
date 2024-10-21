#include <stdint.h>
#include "hash.h"

#ifndef _SHA3_HASH
  #define _SHA3_HASH

  #ifdef __cplusplus
extern "C" {
  #endif

Hash* icicle_create_sha3_256(uint64_t default_input_chunk_size);
Hash* icicle_create_sha3_512(uint64_t default_input_chunk_size);

  #ifdef __cplusplus
}
  #endif

#endif
