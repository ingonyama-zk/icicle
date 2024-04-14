extern "C" cudaError_t ${FIELD}InitializeDomain(
  ${FIELD}::scalar_t* primitive_root, device_context::DeviceContext& ctx, bool fast_twiddles_mode);

extern "C" cudaError_t ${FIELD}NTTCuda(
  const ${FIELD}::scalar_t* input, int size, NTTDir dir, ntt::NTTConfig<${FIELD}::scalar_t>& config, ${FIELD}::scalar_t* output);

extern "C" cudaError_t ${FIELD}ReleaseDomain(device_context::DeviceContext& ctx);