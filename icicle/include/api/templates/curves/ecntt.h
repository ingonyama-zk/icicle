extern "C" cudaError_t ${CURVE}ECNTTCuda(
  const ${CURVE}::projective_t* input, int size, NTTDir dir, ntt::NTTConfig<${CURVE}::scalar_t>& config, ${CURVE}::projective_t* output);