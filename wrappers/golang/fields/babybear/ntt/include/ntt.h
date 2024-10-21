#include <stdbool.h>

#ifndef _BABYBEAR_NTT_H
  #define _BABYBEAR_NTT_H

  #ifdef __cplusplus
extern "C" {
  #endif

typedef struct scalar_t scalar_t;
typedef struct NTTConfig NTTConfig;
typedef struct NTTInitDomainConfig NTTInitDomainConfig;

int babybear_ntt(const scalar_t* input, int size, int dir, NTTConfig* config, scalar_t* output);
int babybear_ntt_init_domain(scalar_t* primitive_root, NTTInitDomainConfig* ctx);
int babybear_ntt_release_domain();
int* babybear_get_root_of_unity(size_t size);

  #ifdef __cplusplus
}
  #endif

#endif