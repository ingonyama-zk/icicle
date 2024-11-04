extern "C" cudaError_t ${FIELD}_extension_ntt_cuda(
  const ${FIELD}::extension_t* input, int size, ntt::NTTDir dir, ntt::NTTConfig<${FIELD}::scalar_t>& config, ${FIELD}::extension_t* output);