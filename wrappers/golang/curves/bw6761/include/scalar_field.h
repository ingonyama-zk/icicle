#include <cuda_runtime.h>
#include "../../include/types.h"
#include <stdbool.h>

#ifndef _BW6_761_FIELD_H
#define _BW6_761_FIELD_H

#ifdef __cplusplus
extern "C" {
#endif

void bw6_761GenerateScalars(scalar_t* scalars, int size);
cudaError_t bw6_761ScalarConvertMontgomery(scalar_t* d_inout, size_t n, bool is_into, DeviceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
