#include <stdbool.h>

#ifndef _STARK252_FIELD_H
#define _STARK252_FIELD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct VecOpsConfig VecOpsConfig;

void stark252_generate_random(scalar_t* scalars, int size);
int stark252_scalar_convert_montgomery(const scalar_t* d_in, size_t n, bool is_into, const VecOpsConfig* ctx, scalar_t* d_out);
void stark252_add(const scalar_t* a, const scalar_t* b, scalar_t* result);
void stark252_sub(const scalar_t* a, const scalar_t* b, scalar_t* result);
void stark252_mul(const scalar_t* a, const scalar_t* b, scalar_t* result);
void stark252_inv(const scalar_t* a, scalar_t* result);
void stark252_pow(const scalar_t* a, int exp, scalar_t* result);

#ifdef __cplusplus
}
#endif

#endif
