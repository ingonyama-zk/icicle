#include "icicle/msm.h"
#include "icicle/dispatcher.h"
#include "icicle/curves/curve_config.h"

using namespace curve_config;

namespace icicle {

  /*************************** MSM ***************************/
  ICICLE_DISPATCHER_INST(MsmDispatcher, msm, MsmImpl);

  extern "C" eIcicleError CONCAT_EXPAND(CURVE, msm)(
    const scalar_t* scalars, const affine_t* bases, int msm_size, const MSMConfig& config, projective_t* results)
  {
    return MsmDispatcher::execute(scalars, bases, msm_size, config, results);
  }

  template <>
  eIcicleError
  msm(const scalar_t* scalars, const affine_t* bases, int msm_size, const MSMConfig& config, projective_t* results)
  {
    return CONCAT_EXPAND(CURVE, msm)(scalars, bases, msm_size, config, results);
  }

  /*************************** MSM PRECOMPUTE BASES ***************************/
  ICICLE_DISPATCHER_INST(MsmPreComputeDispatcher, msm_precompute_bases, MsmPreComputeImpl);

  extern "C" eIcicleError CONCAT_EXPAND(CURVE, msm_precompute_bases)(
    const affine_t* input_bases,
    int nof_bases,
    int precompute_factor,
    const MsmPreComputeConfig& config,
    affine_t* output_bases)
  {
    return MsmPreComputeDispatcher::execute(input_bases, nof_bases, precompute_factor, config, output_bases);
  }

  template <>
  eIcicleError msm_precompute_bases(
    const affine_t* input_bases,
    int nof_bases,
    int precompute_factor,
    const MsmPreComputeConfig& config,
    affine_t* output_bases)
  {
    return CONCAT_EXPAND(CURVE, msm_precompute_bases)(input_bases, nof_bases, precompute_factor, config, output_bases);
  }

} // namespace icicle