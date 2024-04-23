#include <cuda_runtime.h>
#include <stdbool.h>

#ifndef _BW6_761_FIELD_H
#define _BW6_761_FIELD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scalar_t scalar_t;
typedef struct DeviceContext DeviceContext;

void bw6_761_generate_scalars(scalar_t* scalars, int size);
cudaError_t bw6_761_scalar_convert_montgomery(scalar_t* d_inout, size_t n, bool is_into, DeviceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
