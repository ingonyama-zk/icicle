#include "cuda_msm.cuh"
#include "error_translation.h"

/************************************** BACKEND REGISTRATION **************************************/

template <typename S, typename A, typename P>
eIcicleError
msm_cuda(const Device& device, const S* scalars, const A* bases, int msm_size, const MSMConfig& config, P* results)
{
  auto err = msm::msm_cuda(scalars, bases, msm_size, config, results);
  return translateCudaError(err);
}

template <typename A, typename P>
eIcicleError msm_precompute_bases_cuda(
  const Device& device, const A* input_bases, int nof_bases, const MSMConfig& config, A* output_bases)
{
  auto err = msm::cuda_precompute_msm_points<A, P>(input_bases, nof_bases, config, output_bases);
  return translateCudaError(err);
}

REGISTER_MSM_BACKEND("CUDA", (msm_cuda<scalar_t, affine_t, projective_t>));
REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND("CUDA", (msm_precompute_bases_cuda<affine_t, projective_t>));
#ifdef G2
REGISTER_MSM_G2_BACKEND("CUDA", (msm_cuda<scalar_t, g2_affine_t, g2_projective_t>));
REGISTER_MSM_G2_PRE_COMPUTE_BASES_BACKEND("CUDA", (msm_precompute_bases_cuda<g2_affine_t, g2_projective_t>));
#endif // G2