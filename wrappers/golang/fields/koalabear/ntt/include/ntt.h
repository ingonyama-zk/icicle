#include <stdbool.h>

#ifndef _KOALABEAR_NTT_H
#define _KOALABEAR_NTT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct NTTConfig NTTConfig;
typedef struct NTTInitDomainConfig NTTInitDomainConfig;

int koalabear_ntt(const scalar_t* input, int size, int dir, NTTConfig* config, scalar_t* output);
int koalabear_ntt_init_domain(scalar_t* primitive_root, NTTInitDomainConfig* ctx);
int koalabear_ntt_release_domain();
int* koalabear_get_root_of_unity(size_t size);

#ifdef __cplusplus
}
#endif

#endif