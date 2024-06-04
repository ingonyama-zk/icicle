
#include "icicle/msm.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"

#include "icicle/curves/projective.h"
#include "icicle/curves/curve_config.h"

using namespace curve_config;
using namespace icicle;

template <typename S, typename A, typename P>
eIcicleError
cpu_msm(const Device& device, const S* scalars, const A* bases, int msm_size, const MSMConfig& config, P* results)
{
  P res = projective_t::zero();
  for (auto i = 0; i < msm_size; ++i) {
    res = res + P::from_affine(bases[i]) * scalars[i];
  }
  results[0] = res;
  return eIcicleError::SUCCESS;
}

template <typename A>
eIcicleError cpu_msm_precompute_bases(
  const Device& device,
  const A* input_bases,
  int nof_bases,
  int precompute_factor,
  const MsmPreComputeConfig& config,
  A* output_bases)
{
  return eIcicleError::API_NOT_IMPLEMENTED;
}

REGISTER_MSM_BACKEND("CPU", (cpu_msm<scalar_t, affine_t, projective_t>));
REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND("CPU", cpu_msm_precompute_bases<affine_t>);