#pragma once

#ifndef ${CURVE_UPPER}_MSM_API_H
#define ${CURVE_UPPER}_MSM_API_H

#include "curves/params/${CURVE}.cuh"
#include <cuda_runtime.h>
#include "msm/msm.cuh"
#include "gpu-utils/device_context.cuh"

extern "C" cudaError_t ${CURVE}PrecomputeMSMBases(
  ${CURVE}::affine_t* bases,
  int bases_size,
  int precompute_factor,
  int _c,
  bool are_bases_on_device,
  device_context::DeviceContext& ctx,
  ${CURVE}::affine_t* output_bases);

extern "C" cudaError_t ${CURVE}MSMCuda(
  const ${CURVE}::scalar_t* scalars, const ${CURVE}::affine_t* points, int msm_size, msm::MSMConfig& config, ${CURVE}::projective_t* out);

#endif