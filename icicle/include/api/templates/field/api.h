#pragma once

#ifndef ${FIELD_UPPER}_API_H
#define ${FIELD_UPPER}_API_H

#include "${FIELD_HEADER}"
#include <cuda_runtime.h>
#include "gpu-utils/device_context.cuh"

extern "C" void ${FIELD}GenerateScalars(${FIELD}::scalar_t* scalars, int size);

extern "C" cudaError_t ${FIELD}ScalarConvertMontgomery(
  ${FIELD}::scalar_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx);

#endif