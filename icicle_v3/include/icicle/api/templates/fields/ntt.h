extern "C" eIcicleError ${FIELD}_ntt_init_domain(
  ${FIELD}::scalar_t* primitive_root, const NTTInitDomainConfig& config);

extern "C" eIcicleError ${FIELD}_ntt(
  const ${FIELD}::scalar_t* input, int size, NTTDir dir, NTTConfig<${FIELD}::scalar_t>& config, ${FIELD}::scalar_t* output);

extern "C" eIcicleError ${FIELD}_ntt_release_domain();