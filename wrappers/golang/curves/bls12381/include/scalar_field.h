#include <cuda_runtime.h>
#include "../../include/types.h"
#include <stdbool.h>

#ifndef _BLS12_381_FIELD_H
#define _BLS12_381_FIELD_H

#ifdef __cplusplus
extern "C" {
#endif

void bls12_381GenerateScalars(scalar_t* scalars, int size);
cudaError_t bls12_381ScalarConvertMontgomery(scalar_t* d_inout, size_t n, bool is_into, DeviceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
