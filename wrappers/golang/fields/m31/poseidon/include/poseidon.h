#include <stdint.h>

#ifndef _M31_POSEIDON
  #define _M31_POSEIDON

  #ifdef __cplusplus
extern "C" {
  #endif

typedef struct scalar_t scalar_t;
typedef struct Hash Hash;

Hash* m31_create_poseidon_hasher(unsigned t, const scalar_t* domain_tag, unsigned input_size);

  #ifdef __cplusplus
}
  #endif

#endif
