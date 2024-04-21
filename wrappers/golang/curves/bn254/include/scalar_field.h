#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _BN254_FIELD_H
#define _BN254_FIELD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct DeviceContext DeviceContext;

void bn254_generate_scalars(scalar_t* scalars, int size);
cudaError_t bn254_scalar_convert_montgomery(scalar_t* d_inout, size_t n, bool is_into, DeviceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
