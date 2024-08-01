#include <stdbool.h>

#ifndef _BLS12_377_FIELD_H
#define _BLS12_377_FIELD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct VecOpsConfig VecOpsConfig;

void bls12_377_generate_scalars(scalar_t* scalars, int size);
int bls12_377_scalar_convert_montgomery(const scalar_t* d_in, size_t n, bool is_into, const VecOpsConfig* ctx, scalar_t* d_out);

#ifdef __cplusplus
}
#endif

#endif
