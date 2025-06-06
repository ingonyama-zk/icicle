#include <stdint.h>

#ifndef _BLS12_381_POSEIDON
  #define _BLS12_381_POSEIDON

  #ifdef __cplusplus
extern "C" {
  #endif

typedef struct scalar_t scalar_t;
typedef struct Hash Hash;

Hash* bls12_381_create_poseidon_hasher(unsigned t, const scalar_t* domain_tag, unsigned input_size);

  #ifdef __cplusplus
}
  #endif

#endif
