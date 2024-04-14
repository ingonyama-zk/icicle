#pragma once

#ifndef ${CURVE_UPPER}_CURVE_G2_API_H
#define ${CURVE_UPPER}_CURVE_G2_API_H

#include "curves/params/${CURVE}.cuh"
#include <cuda_runtime.h>
#include "gpu-utils/device_context.cuh"

extern "C" bool ${CURVE}G2Eq(${CURVE}::g2_projective_t* point1, ${CURVE}::g2_projective_t* point2);

extern "C" void ${CURVE}G2ToAffine(${CURVE}::g2_projective_t* point, ${CURVE}::g2_affine_t* point_out);

extern "C" void ${CURVE}G2GenerateProjectivePoints(${CURVE}::g2_projective_t* points, int size);

extern "C" void ${CURVE}G2GenerateAffinePoints(${CURVE}::g2_affine_t* points, int size);

extern "C" cudaError_t ${CURVE}G2AffineConvertMontgomery(
  ${CURVE}::g2_affine_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx);

extern "C" cudaError_t ${CURVE}G2ProjectiveConvertMontgomery(
  ${CURVE}::g2_projective_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx);

#endif