#pragma once

#ifndef ${CURVE_UPPER}_CURVE_API_H
#define ${CURVE_UPPER}_CURVE_API_H

#include "curves/params/${CURVE}.cuh"
#include <cuda_runtime.h>
#include "gpu-utils/device_context.cuh"

extern "C" bool ${CURVE}Eq(${CURVE}::projective_t* point1, ${CURVE}::projective_t* point2);

extern "C" void ${CURVE}ToAffine(${CURVE}::projective_t* point, ${CURVE}::affine_t* point_out);

extern "C" void ${CURVE}GenerateProjectivePoints(${CURVE}::projective_t* points, int size);

extern "C" void ${CURVE}GenerateAffinePoints(${CURVE}::affine_t* points, int size);

extern "C" cudaError_t ${CURVE}AffineConvertMontgomery(
  ${CURVE}::affine_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx);

extern "C" cudaError_t ${CURVE}ProjectiveConvertMontgomery(
  ${CURVE}::projective_t* d_inout, size_t n, bool is_into, device_context::DeviceContext& ctx)

#endif