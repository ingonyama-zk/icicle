#pragma once

#ifndef ${FIELD_UPPER}_VEC_OPS_API_H
#define ${FIELD_UPPER}_VEC_OPS_API_H

#include "${FIELD_HEADER}"
#include <cuda_runtime.h>
#include "vec_ops/vec_ops.cuh"
#include "gpu-utils/device_context.cuh"

extern "C" cudaError_t ${FIELD}MulCuda(
  ${FIELD}::scalar_t* vec_a, ${FIELD}::scalar_t* vec_b, int n, VecOpsConfig<${FIELD}::scalar_t>& config, ${FIELD}::scalar_t* result);

extern "C" cudaError_t ${FIELD}AddCuda(
  ${FIELD}::scalar_t* vec_a, ${FIELD}::scalar_t* vec_b, int n, VecOpsConfig<${FIELD}::scalar_t>& config, ${FIELD}::scalar_t* result);

extern "C" cudaError_t ${FIELD}SubCuda(
  ${FIELD}::scalar_t* vec_a, ${FIELD}::scalar_t* vec_b, int n, VecOpsConfig<${FIELD}::scalar_t>& config, ${FIELD}::scalar_t* result);

extern "C" cudaError_t ${FIELD}TransposeMatrix(
  const ${FIELD}::scalar_t* input,
  uint32_t row_size,
  uint32_t column_size,
  ${FIELD}::scalar_t* output,
  device_context::DeviceContext& ctx,
  bool on_device,
  bool is_async);

#endif