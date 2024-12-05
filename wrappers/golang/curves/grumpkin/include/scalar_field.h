#include <stdbool.h>

#ifndef _GRUMPKIN_FIELD_H
#define _GRUMPKIN_FIELD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct VecOpsConfig VecOpsConfig;

void grumpkin_generate_scalars(scalar_t* scalars, int size);
int grumpkin_scalar_convert_montgomery(const scalar_t* d_in, size_t n, bool is_into, const VecOpsConfig* ctx, scalar_t* d_out);
void grumpkin_add(const scalar_t* a, const scalar_t* b, scalar_t* result);
void grumpkin_sub(const scalar_t* a, const scalar_t* b, scalar_t* result);
void grumpkin_mul(const scalar_t* a, const scalar_t* b, scalar_t* result);
void grumpkin_inv(const scalar_t* a, scalar_t* result);

#ifdef __cplusplus
}
#endif

#endif
