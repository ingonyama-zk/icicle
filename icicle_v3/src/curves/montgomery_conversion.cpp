#include "icicle/curves/montgomery_conversion.h"
#include "icicle/dispatcher.h"
#include "icicle/curves/curve_config.h"

using namespace curve_config;

namespace icicle {

  /*************************** AFFINE CONVERT MONTGOMERY ***************************/
  ICICLE_DISPATCHER_INST(AffineConvertMont, affine_convert_montgomery, AffineConvertMontImpl);

  extern "C" eIcicleError CONCAT_EXPAND(CURVE, affine_convert_montgomery)(
    const affine_t* input, size_t n, bool is_into, const ConvertMontgomeryConfig& config, affine_t* output)
  {
    return AffineConvertMont::execute(input, n, is_into, config, output);
  }

  template <>
  eIcicleError points_convert_montgomery(
    const affine_t* input, size_t n, bool is_into, const ConvertMontgomeryConfig& config, affine_t* output)
  {
    return CONCAT_EXPAND(CURVE, affine_convert_montgomery)(input, n, is_into, config, output);
  }

  /*************************** PROJECTIVE CONVERT MONTGOMERY ***************************/
  ICICLE_DISPATCHER_INST(ProjectiveConvertMont, projective_convert_montgomery, ProjectiveConvertMontImpl);

  extern "C" eIcicleError CONCAT_EXPAND(CURVE, projective_convert_montgomery)(
    const projective_t* input, size_t n, bool is_into, const ConvertMontgomeryConfig& config, projective_t* output)
  {
    return ProjectiveConvertMont::execute(input, n, is_into, config, output);
  }

  template <>
  eIcicleError points_convert_montgomery(
    const projective_t* input, size_t n, bool is_into, const ConvertMontgomeryConfig& config, projective_t* output)
  {
    return CONCAT_EXPAND(CURVE, projective_convert_montgomery)(input, n, is_into, config, output);
  }
} // namespace icicle