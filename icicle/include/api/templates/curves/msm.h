extern "C" cudaError_t ${CURVE} _precompute_msm_bases_cuda(
  $ { CURVE } ::affine_t* bases, int msm_size, msm::MSMConfig& config, $ { CURVE } ::affine_t* output_bases);

extern "C" cudaError_t ${CURVE} _msm_cuda(
  const $ { CURVE } ::scalar_t* scalars,
  const $ { CURVE } ::affine_t* points,
  int msm_size,
  msm::MSMConfig& config,
  $ { CURVE } ::projective_t* out);