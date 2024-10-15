#include <stdbool.h>

#ifndef _BLS12_377_NTT_H
  #define _BLS12_377_NTT_H

  #ifdef __cplusplus
extern "C" {
  #endif

typedef struct scalar_t scalar_t;
typedef struct NTTConfig NTTConfig;
typedef struct NTTInitDomainConfig NTTInitDomainConfig;

int bls12_377_ntt(const scalar_t* input, int size, int dir, NTTConfig* config, scalar_t* output);
int bls12_377_ntt_init_domain(scalar_t* primitive_root, NTTInitDomainConfig* ctx);
int bls12_377_ntt_release_domain();
int* bls12_377_get_root_of_unity(size_t size);

  #ifdef __cplusplus
}
  #endif

#endif