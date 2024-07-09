extern "C" cudaError_t ${FIELD}_extension_mul_cuda(
  ${FIELD}::extension_t* vec_a, ${FIELD}::extension_t* vec_b, int n, vec_ops::VecOpsConfig& config, ${FIELD}::extension_t* result);

extern "C" cudaError_t ${FIELD}_extension_add_cuda(
  ${FIELD}::extension_t* vec_a, ${FIELD}::extension_t* vec_b, int n, vec_ops::VecOpsConfig& config, ${FIELD}::extension_t* result);

extern "C" cudaError_t ${FIELD}_extension_accumulate_cuda(
  ${FIELD}::extension_t* vec_a, ${FIELD}::extension_t* vec_b, int n, vec_ops::VecOpsConfig& config);

extern "C" cudaError_t ${FIELD}_extension_sub_cuda(
  ${FIELD}::extension_t* vec_a, ${FIELD}::extension_t* vec_b, int n, vec_ops::VecOpsConfig& config, ${FIELD}::extension_t* result);

extern "C" cudaError_t ${FIELD}_extension_transpose_matrix_cuda(
  const ${FIELD}::extension_t* input,
  uint32_t row_size,
  uint32_t column_size,
  ${FIELD}::extension_t* output,
  device_context::DeviceContext& ctx,
  bool on_device,
  bool is_async);

extern "C" cudaError_t ${FIELD}_extension_bit_reverse_cuda(
  const ${FIELD}::extension_t* input, uint64_t n, vec_ops::BitReverseConfig& config, ${FIELD}::extension_t* output);
