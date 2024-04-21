#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _BABYBEAR_FIELD_H
#define _BABYBEAR_FIELD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct DeviceContext DeviceContext;

void babybearGenerateScalars(scalar_t* scalars, int size);
cudaError_t babybearScalarConvertMontgomery(scalar_t* d_inout, size_t n, bool is_into, DeviceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
