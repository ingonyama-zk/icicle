extern "C" eIcicleError ${CURVE}_ecntt(
  const ${CURVE}::projective_t* input, int size, NTTDir dir, NTTConfig<${CURVE}::scalar_t>& config, ${CURVE}::projective_t* output);