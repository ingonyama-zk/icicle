#include <stdbool.h>

#ifndef _BW6_761_NTT_H
  #define _BW6_761_NTT_H

  #ifdef __cplusplus
extern "C" {
  #endif

typedef struct scalar_t scalar_t;
typedef struct NTTConfig NTTConfig;
typedef struct NTTInitDomainConfig NTTInitDomainConfig;

int bw6_761_ntt(const scalar_t* input, int size, int dir, NTTConfig* config, scalar_t* output);
int bw6_761_ntt_init_domain(scalar_t* primitive_root, NTTInitDomainConfig* ctx);
int bw6_761_ntt_release_domain();
int* bw6_761_get_root_of_unity(size_t size);

  #ifdef __cplusplus
}
  #endif

#endif