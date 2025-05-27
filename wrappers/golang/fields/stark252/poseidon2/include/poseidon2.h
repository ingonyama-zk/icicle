#include <stdint.h>

#ifndef _STARK252_POSEIDON2
  #define _STARK252_POSEIDON2

  #ifdef __cplusplus
extern "C" {
  #endif

typedef struct scalar_t scalar_t;
typedef struct Hash Hash;

Hash* stark252_create_poseidon2_hasher(unsigned t, const scalar_t* domain_tag, unsigned input_size);

  #ifdef __cplusplus
}
  #endif

#endif
