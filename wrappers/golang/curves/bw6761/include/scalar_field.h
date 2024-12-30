#include <stdbool.h>

#ifndef _BW6_761_FIELD_H
#define _BW6_761_FIELD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct VecOpsConfig VecOpsConfig;

void bw6_761_generate_scalars(scalar_t* scalars, int size);
int bw6_761_scalar_convert_montgomery(const scalar_t* d_in, size_t n, bool is_into, const VecOpsConfig* ctx, scalar_t* d_out);
void bw6_761_add(const scalar_t* a, const scalar_t* b, scalar_t* result);
void bw6_761_sub(const scalar_t* a, const scalar_t* b, scalar_t* result);
void bw6_761_mul(const scalar_t* a, const scalar_t* b, scalar_t* result);
void bw6_761_inv(const scalar_t* a, scalar_t* result);
void bw6_761_pow(const scalar_t* a, int exp, scalar_t* result);

#ifdef __cplusplus
}
#endif

#endif
