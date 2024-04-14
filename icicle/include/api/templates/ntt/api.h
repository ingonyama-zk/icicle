#pragma once

#ifndef ${FIELD_UPPER}_NTT_API_H
#define ${FIELD_UPPER}_NTT_API_H

#include "${FIELD_HEADER}"
#include <cuda_runtime.h>
#include "ntt/ntt.cuh"
#include "gpu-utils/device_context.cuh"

extern "C" cudaError_t ${FIELD}InitializeDomain(
  ${FIELD}::scalar_t* primitive_root, device_context::DeviceContext& ctx, bool fast_twiddles_mode);

extern "C" cudaError_t ${FIELD}NTTCuda(
  const ${FIELD}::scalar_t* input, int size, NTTDir dir, ntt::NTTConfig<${FIELD}::scalar_t>& config, ${FIELD}::scalar_t* output)

extern "C" cudaError_t ${FIELD}ReleaseDomain(device_context::DeviceContext& ctx);

#endif