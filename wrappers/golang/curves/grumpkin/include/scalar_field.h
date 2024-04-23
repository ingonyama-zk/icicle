#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _GRUMPKIN_FIELD_H
#define _GRUMPKIN_FIELD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct DeviceContext DeviceContext;

void grumpkin_generate_scalars(scalar_t* scalars, int size);
cudaError_t grumpkin_scalar_convert_montgomery(scalar_t* d_inout, size_t n, bool is_into, DeviceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
