#include <stdbool.h>

#ifndef _STARK252_NTT_H
#define _STARK252_NTT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct NTTConfig NTTConfig;
typedef struct NTTInitDomainConfig NTTInitDomainConfig;

int stark252_ntt(const scalar_t* input, int size, int dir, NTTConfig* config, scalar_t* output);
int stark252_ntt_init_domain(scalar_t* primitive_root, NTTInitDomainConfig* ctx);
int stark252_ntt_release_domain();
int stark252_get_root_of_unity(size_t size, scalar_t* output);

#ifdef __cplusplus
}
#endif

#endif