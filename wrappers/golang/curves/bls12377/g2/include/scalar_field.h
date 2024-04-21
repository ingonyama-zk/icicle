#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _BLS12_377_FIELD_H
#define _BLS12_377_FIELD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct DeviceContext DeviceContext;

void bls12_377GenerateScalars(scalar_t* scalars, int size);
cudaError_t bls12_377ScalarConvertMontgomery(scalar_t* d_inout, size_t n, bool is_into, DeviceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
