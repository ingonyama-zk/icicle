extern "C" cudaError_t ${CURVE}_g2_precompute_msm_bases_cuda(
  ${CURVE}::g2_affine_t* bases,
  int bases_size,
  int precompute_factor,
  int _c,
  bool are_bases_on_device,
  device_context::DeviceContext& ctx,
  ${CURVE}::g2_affine_t* output_bases);

extern "C" cudaError_t ${CURVE}_g2_msm_cuda(
  const ${CURVE}::scalar_t* scalars, const ${CURVE}::g2_affine_t* points, int msm_size, msm::MSMConfig& config, ${CURVE}::g2_projective_t* out);