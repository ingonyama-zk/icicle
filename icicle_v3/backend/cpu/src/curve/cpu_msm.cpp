
#include "icicle/msm.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"

#include "icicle/curves/projective.h"
#include "icicle/curves/curve_config.h"

using namespace curve_config;
using namespace icicle;

eIcicleError cpu_msm(
  const Device& device,
  const scalar_t* scalars,
  const affine_t* bases,
  int msm_size,
  const MSMConfig& config,
  ResType* results)
{
  projective_t res = projective_t::zero();
  for (auto i = 0; i < msm_size; ++i) {
    res = res + projective_t::from_affine(bases[i]) * scalars[i];
  }
  results[0] =
    projective_t::to_affine(res); // TODO need to solve the weird linkage issue when output type is projective
  return eIcicleError::SUCCESS;
}

REGISTER_MSM_BACKEND("CPU", cpu_msm);