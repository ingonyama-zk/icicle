extern "C" cudaError_t ${FIELD}ExtensionMulCuda(
  ${FIELD}::extension_t* vec_a, ${FIELD}::extension_t* vec_b, int n, vec_ops::VecOpsConfig<${FIELD}::extension_t>& config, ${FIELD}::extension_t* result);

extern "C" cudaError_t ${FIELD}ExtensionAddCuda(
  ${FIELD}::extension_t* vec_a, ${FIELD}::extension_t* vec_b, int n, vec_ops::VecOpsConfig<${FIELD}::extension_t>& config, ${FIELD}::extension_t* result);

extern "C" cudaError_t ${FIELD}ExtensionSubCuda(
  ${FIELD}::extension_t* vec_a, ${FIELD}::extension_t* vec_b, int n, vec_ops::VecOpsConfig<${FIELD}::extension_t>& config, ${FIELD}::extension_t* result);

extern "C" cudaError_t ${FIELD}ExtensionTransposeMatrix(
  const ${FIELD}::extension_t* input,
  uint32_t row_size,
  uint32_t column_size,
  ${FIELD}::extension_t* output,
  device_context::DeviceContext& ctx,
  bool on_device,
  bool is_async);