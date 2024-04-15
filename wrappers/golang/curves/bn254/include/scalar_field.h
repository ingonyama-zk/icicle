#include <cuda_runtime.h>
#include "../../include/types.h"
#include <stdbool.h>

#ifndef _BN254_FIELD_H
#define _BN254_FIELD_H

#ifdef __cplusplus
extern "C" {
#endif

void bn254GenerateScalars(scalar_t* scalars, int size);
cudaError_t bn254ScalarConvertMontgomery(scalar_t* d_inout, size_t n, bool is_into, DeviceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
