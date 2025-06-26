#include <stdbool.h>

#ifndef _M31_FIELD_H
#define _M31_FIELD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct VecOpsConfig VecOpsConfig;

void m31_generate_scalars(scalar_t* scalars, int size);
int m31_scalar_convert_montgomery(const scalar_t* d_in, size_t n, bool is_into, const VecOpsConfig* ctx, scalar_t* d_out);
void m31_add(const scalar_t* a, const scalar_t* b, scalar_t* result);
void m31_sub(const scalar_t* a, const scalar_t* b, scalar_t* result);
void m31_mul(const scalar_t* a, const scalar_t* b, scalar_t* result);
void m31_inv(const scalar_t* a, scalar_t* result);
void m31_pow(const scalar_t* a, int exp, scalar_t* result);

#ifdef __cplusplus
}
#endif

#endif
