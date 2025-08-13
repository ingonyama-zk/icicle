#include <stdbool.h>

#ifndef _M31_EXTENSION_FIELD_H
#define _M31_EXTENSION_FIELD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct VecOpsConfig VecOpsConfig;

void m31_extension_generate_random(scalar_t* scalars, int size);
int m31_extension_scalar_convert_montgomery(const scalar_t* d_in, size_t n, bool is_into, const VecOpsConfig* ctx, scalar_t* d_out);
void m31_extension_add(const scalar_t* a, const scalar_t* b, scalar_t* result);
void m31_extension_sub(const scalar_t* a, const scalar_t* b, scalar_t* result);
void m31_extension_mul(const scalar_t* a, const scalar_t* b, scalar_t* result);
void m31_extension_inv(const scalar_t* a, scalar_t* result);
void m31_extension_pow(const scalar_t* a, int exp, scalar_t* result);

#ifdef __cplusplus
}
#endif

#endif
