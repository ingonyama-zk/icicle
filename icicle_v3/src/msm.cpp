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

#ifdef G2
  ICICLE_DISPATCHER_INST(MsmG2Dispatcher, g2_msm, MsmG2Impl);

  extern "C" eIcicleError CONCAT_EXPAND(CURVE, g2_msm)(
    const scalar_t* scalars, const g2_affine_t* bases, int msm_size, const MSMConfig& config, g2_projective_t* results)
  {
    return MsmG2Dispatcher::execute(scalars, bases, msm_size, config, results);
  }

  template <>
  eIcicleError msm(
    const scalar_t* scalars, const g2_affine_t* bases, int msm_size, const MSMConfig& config, g2_projective_t* results)
  {
    return CONCAT_EXPAND(CURVE, g2_msm)(scalars, bases, msm_size, config, results);
  }
#endif // G2

  /*************************** MSM PRECOMPUTE BASES ***************************/
  ICICLE_DISPATCHER_INST(MsmPreComputeDispatcher, msm_precompute_bases, MsmPreComputeImpl);

  extern "C" eIcicleError CONCAT_EXPAND(CURVE, msm_precompute_bases)(
    const affine_t* input_bases, int nof_bases, const MSMConfig& config, affine_t* output_bases)
  {
    return MsmPreComputeDispatcher::execute(input_bases, nof_bases, config, output_bases);
  }

  template <>
  eIcicleError
  msm_precompute_bases(const affine_t* input_bases, int nof_bases, const MSMConfig& config, affine_t* output_bases)
  {
    return CONCAT_EXPAND(CURVE, msm_precompute_bases)(input_bases, nof_bases, config, output_bases);
  }

#ifdef G2
  ICICLE_DISPATCHER_INST(MsmG2PreComputeDispatcher, g2_msm_precompute_bases, MsmG2PreComputeImpl);

  extern "C" eIcicleError CONCAT_EXPAND(CURVE, g2_msm_precompute_bases)(
    const g2_affine_t* input_bases, int nof_bases, const MSMConfig& config, g2_affine_t* output_bases)
  {
    return MsmG2PreComputeDispatcher::execute(input_bases, nof_bases, config, output_bases);
  }

#ifndef G1_AFFINE_SAME_TYPE_AS_G2_AFFINE
  template <>
  eIcicleError msm_precompute_bases(
    const g2_affine_t* input_bases, int nof_bases, const MSMConfig& config, g2_affine_t* output_bases)
  {
    return CONCAT_EXPAND(CURVE, g2_msm_precompute_bases)(input_bases, nof_bases, config, output_bases);
  }
#endif // !G1_AFFINE_SAME_TYPE_AS_G2_AFFINE
#endif // G2

} // namespace icicle