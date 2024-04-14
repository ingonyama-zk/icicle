#pragma once

#ifndef ${CURVE_UPPER}_MSM_G2_API_H
#define ${CURVE_UPPER}_MSM_G2_API_H

#include "curves/params/${CURVE}.cuh"
#include <cuda_runtime.h>
#include "msm/msm.cuh"
#include "gpu-utils/device_context.cuh"

extern "C" cudaError_t ${CURVE}G2PrecomputeMSMBases(
  ${CURVE}::g2_affine_t* bases,
  int bases_size,
  int precompute_factor,
  int _c,
  bool are_bases_on_device,
  device_context::DeviceContext& ctx,
  ${CURVE}::g2_affine_t* output_bases);

extern "C" cudaError_t ${CURVE}G2MSMCuda(
  const ${CURVE}::scalar_t* scalars, const ${CURVE}::g2_affine_t* points, int msm_size, msm::MSMConfig& config, ${CURVE}::g2_projective_t* out);

#endif