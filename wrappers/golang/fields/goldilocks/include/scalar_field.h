#include <stdbool.h>

#ifndef _GOLDILOCKS_FIELD_H
#define _GOLDILOCKS_FIELD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct VecOpsConfig VecOpsConfig;

void goldilocks_generate_random(scalar_t* scalars, int size);
int goldilocks_scalar_convert_montgomery(const scalar_t* d_in, size_t n, bool is_into, const VecOpsConfig* ctx, scalar_t* d_out);
void goldilocks_add(const scalar_t* a, const scalar_t* b, scalar_t* result);
void goldilocks_sub(const scalar_t* a, const scalar_t* b, scalar_t* result);
void goldilocks_mul(const scalar_t* a, const scalar_t* b, scalar_t* result);
void goldilocks_inv(const scalar_t* a, scalar_t* result);
void goldilocks_pow(const scalar_t* a, int exp, scalar_t* result);

#ifdef __cplusplus
}
#endif

#endif
