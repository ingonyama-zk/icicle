#include "icicle/msm.h"
#include "icicle/dispatcher.h"
#include "icicle/curves/curve_config.h"

using namespace curve_config;

namespace icicle {

  /*************************** MSM ***************************/
  ICICLE_DISPATCHER_INST(MsmDispatcher, msm, MsmImpl);

  extern "C" eIcicleError CONCAT_EXPAND(CURVE, msm)(
    const scalar_t* scalars, const affine_t* bases, int msm_size, const MSMConfig& config, ResType* results)
  {
    return MsmDispatcher::execute(scalars, bases, msm_size, config, results);
  }

  template <>
  eIcicleError
  msm(const scalar_t* scalars, const affine_t* bases, int msm_size, const MSMConfig& config, ResType* results)
  {
    return CONCAT_EXPAND(CURVE, msm)(scalars, bases, msm_size, config, results);
  }

} // namespace icicle