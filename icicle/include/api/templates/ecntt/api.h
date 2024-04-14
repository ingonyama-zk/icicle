#pragma once

#ifndef ${CURVE_UPPER}_ECNTT_API_H
#define ${CURVE_UPPER}_ECNTT_API_H

#include "curves/params/${CURVE}.cuh"
#include <cuda_runtime.h>
#include "gpu-utils/device_context.cuh"
#include "ntt/ntt.cuh"

extern "C" cudaError_t ${CURVE}ECNTTCuda(
  const ${CURVE}::projective_t* input, int size, NTTDir dir, ntt::NTTConfig<${CURVE}::scalar_t>& config, ${CURVE}::projective_t* output);

#endif