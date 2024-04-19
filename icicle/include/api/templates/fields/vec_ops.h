extern "C" cudaError_t ${FIELD}MulCuda(
  ${FIELD}::scalar_t* vec_a, ${FIELD}::scalar_t* vec_b, int n, vec_ops::VecOpsConfig& config, ${FIELD}::scalar_t* result);

extern "C" cudaError_t ${FIELD}AddCuda(
  ${FIELD}::scalar_t* vec_a, ${FIELD}::scalar_t* vec_b, int n, vec_ops::VecOpsConfig& config, ${FIELD}::scalar_t* result);

extern "C" cudaError_t ${FIELD}SubCuda(
  ${FIELD}::scalar_t* vec_a, ${FIELD}::scalar_t* vec_b, int n, vec_ops::VecOpsConfig& config, ${FIELD}::scalar_t* result);

extern "C" cudaError_t ${FIELD}TransposeMatrix(
  const ${FIELD}::scalar_t* input,
  uint32_t row_size,
  uint32_t column_size,
  ${FIELD}::scalar_t* output,
  device_context::DeviceContext& ctx,
  bool on_device,
  bool is_async);