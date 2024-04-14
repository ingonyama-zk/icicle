extern "C" cudaError_t ${FIELD}ExtensionNTTCuda(
  const ${FIELD}::extension_t* input, int size, ntt::NTTDir dir, ntt::NTTConfig<${FIELD}::scalar_t>& config, ${FIELD}::extension_t* output);