extern "C" cudaError_t ${FIELD} _initialize_domain(
  $ { FIELD } ::scalar_t* primitive_root, device_context::DeviceContext& ctx, bool fast_twiddles_mode);

extern "C" cudaError_t ${FIELD} _ntt_cuda(
  const $ { FIELD } ::scalar_t* input,
  int size,
  ntt::NTTDir dir,
  ntt::NTTConfig<$ { FIELD } ::scalar_t>& config,
  $ { FIELD } ::scalar_t* output);

extern "C" cudaError_t ${FIELD} _release_domain(device_context::DeviceContext& ctx);