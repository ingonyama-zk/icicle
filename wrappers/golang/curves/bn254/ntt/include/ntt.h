#include <stdbool.h>

#ifndef _BN254_NTT_H
  #define _BN254_NTT_H

  #ifdef __cplusplus
extern "C" {
  #endif

typedef struct scalar_t scalar_t;
typedef struct NTTConfig NTTConfig;
typedef struct NTTInitDomainConfig NTTInitDomainConfig;

int bn254_ntt(const scalar_t* input, int size, int dir, NTTConfig* config, scalar_t* output);
int bn254_ntt_init_domain(scalar_t* primitive_root, NTTInitDomainConfig* ctx);
int bn254_ntt_release_domain();
int* bn254_get_root_of_unity(size_t size);

  #ifdef __cplusplus
}
  #endif

#endif