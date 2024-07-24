
#include "icicle/backend/msm_backend.h"
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
  for (auto batch_idx = 0; batch_idx < config.batch_size; ++batch_idx) {
    P res = P::zero();
    const S* batch_scalars = scalars + msm_size * batch_idx;
    const A* batch_bases = config.are_bases_shared ? bases : bases + msm_size * batch_idx;
    for (auto i = 0; i < msm_size; ++i) {
      res = res + P::from_affine(batch_bases[i]) * batch_scalars[i];
    }
    results[batch_idx] = res;
  }
  return eIcicleError::SUCCESS;
}

template <typename A>
eIcicleError cpu_msm_precompute_bases(
  const Device& device, const A* input_bases, int nof_bases, const MSMConfig& config, A* output_bases)
{
  ICICLE_ASSERT(!config.are_points_on_device && !config.are_scalars_on_device);
  memcpy(output_bases, input_bases, sizeof(A) * nof_bases);
  return eIcicleError::SUCCESS;
}

REGISTER_MSM_BACKEND("CPU", (cpu_msm<scalar_t, affine_t, projective_t>));
REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND("CPU", cpu_msm_precompute_bases<affine_t>);

#ifdef G2
REGISTER_MSM_G2_BACKEND("CPU", (cpu_msm<scalar_t, g2_affine_t, g2_projective_t>));
REGISTER_MSM_G2_PRE_COMPUTE_BASES_BACKEND("CPU", cpu_msm_precompute_bases<g2_affine_t>);
#endif // G2