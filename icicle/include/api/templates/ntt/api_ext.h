#pragma once

#ifndef ${FIELD_UPPER}_NTT_EXT_API_H
#define ${FIELD_UPPER}_NTT_EXT_API_H

#include "${FIELD_HEADER}"
#include <cuda_runtime.h>
#include "ntt/ntt.cuh"
#include "gpu-utils/device_context.cuh"

extern "C" cudaError_t ${FIELD}ExtensionNTTCuda(
  const ${FIELD}::extension_t* input, int size, NTTDir dir, ntt::NTTConfig<${FIELD}::scalar_t>& config, ${FIELD}::extension_t* output)

#endif