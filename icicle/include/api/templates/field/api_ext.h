#pragma once

#ifndef ${FIELD_UPPER}_EXT_API_H
#define ${FIELD_UPPER}_EXT_API_H

#include "${FIELD_HEADER}"
#include <cuda_runtime.h>
#include "gpu-utils/device_context.cuh"

extern "C" void ${FIELD}ExtensionGenerateScalars(${FIELD}::extension_t* scalars, int size);

extern "C" cudaError_t ${FIELD}ExtensionScalarConvertMontgomery(
  ${FIELD}::extension_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx);

#endif