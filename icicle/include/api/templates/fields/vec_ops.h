extern "C" cudaError_t ${FIELD}_mul_cuda(
  ${FIELD}::scalar_t* vec_a, ${FIELD}::scalar_t* vec_b, int n, vec_ops::VecOpsConfig& config, ${FIELD}::scalar_t* result);

extern "C" cudaError_t ${FIELD}_add_cuda(
  ${FIELD}::scalar_t* vec_a, ${FIELD}::scalar_t* vec_b, int n, vec_ops::VecOpsConfig& config, ${FIELD}::scalar_t* result);

extern "C" cudaError_t ${FIELD}_sub_cuda(
  ${FIELD}::scalar_t* vec_a, ${FIELD}::scalar_t* vec_b, int n, vec_ops::VecOpsConfig& config, ${FIELD}::scalar_t* result);

extern "C" cudaError_t ${FIELD}_transpose_matrix_cuda(
  const ${FIELD}::scalar_t* input,
  uint32_t row_size,
  uint32_t column_size,
  ${FIELD}::scalar_t* output,
  device_context::DeviceContext& ctx,
  bool on_device,
  bool is_async);

extern "C" cudaError_t ${FIELD}_bit_reverse_cuda(
  const ${FIELD}::scalar_t* input,
  uint64_t n,
  vec_ops::BitReverseConfig& config,
  ${FIELD}::scalar_t* output);