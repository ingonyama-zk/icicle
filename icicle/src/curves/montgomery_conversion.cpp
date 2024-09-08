#include "icicle/vec_ops.h"
#include "icicle/curves/montgomery_conversion.h"
#include "icicle/dispatcher.h"
#include "icicle/curves/curve_config.h"

using namespace curve_config;

namespace icicle {

  /*************************** AFFINE CONVERT MONTGOMERY ***************************/
  ICICLE_DISPATCHER_INST(AffineConvertMont, affine_convert_montgomery, AffineConvertMontImpl);

  extern "C" eIcicleError CONCAT_EXPAND(CURVE, affine_convert_montgomery)(
    const affine_t* input, uint64_t n, bool is_into, const VecOpsConfig* config, affine_t* output)
  {
    return AffineConvertMont::execute(input, n, is_into, *config, output);
  }

  template <>
  eIcicleError
  convert_montgomery(const affine_t* input, uint64_t n, bool is_into, const VecOpsConfig& config, affine_t* output)
  {
    return CONCAT_EXPAND(CURVE, affine_convert_montgomery)(input, n, is_into, &config, output);
  }

#ifdef G2
  ICICLE_DISPATCHER_INST(AffineG2ConvertMont, g2_affine_convert_montgomery, AffineG2ConvertMontImpl);

  extern "C" eIcicleError CONCAT_EXPAND(CURVE, g2_affine_convert_montgomery)(
    const g2_affine_t* input, size_t n, bool is_into, const VecOpsConfig* config, g2_affine_t* output)
  {
    return AffineG2ConvertMont::execute(input, n, is_into, *config, output);
  }

#ifndef G1_AFFINE_SAME_TYPE_AS_G2_AFFINE
  template <>
  eIcicleError convert_montgomery(
    const g2_affine_t* input, uint64_t n, bool is_into, const VecOpsConfig& config, g2_affine_t* output)
  {
    return CONCAT_EXPAND(CURVE, g2_affine_convert_montgomery)(input, n, is_into, &config, output);
  }
#endif //! G1_AFFINE_SAME_TYPE_AS_G2_AFFINE
#endif // G2
  /*************************** PROJECTIVE CONVERT MONTGOMERY ***************************/
  ICICLE_DISPATCHER_INST(ProjectiveConvertMont, projective_convert_montgomery, ProjectiveConvertMontImpl);

  extern "C" eIcicleError CONCAT_EXPAND(CURVE, projective_convert_montgomery)(
    const projective_t* input, uint64_t n, bool is_into, const VecOpsConfig* config, projective_t* output)
  {
    return ProjectiveConvertMont::execute(input, n, is_into, *config, output);
  }

  template <>
  eIcicleError convert_montgomery(
    const projective_t* input, uint64_t n, bool is_into, const VecOpsConfig& config, projective_t* output)
  {
    return CONCAT_EXPAND(CURVE, projective_convert_montgomery)(input, n, is_into, &config, output);
  }

#ifdef G2
  ICICLE_DISPATCHER_INST(ProjectiveG2ConvertMont, g2_projective_convert_montgomery, ProjectiveG2ConvertMontImpl);

  extern "C" eIcicleError CONCAT_EXPAND(CURVE, g2_projective_convert_montgomery)(
    const g2_projective_t* input, uint64_t n, bool is_into, const VecOpsConfig* config, g2_projective_t* output)
  {
    return ProjectiveG2ConvertMont::execute(input, n, is_into, *config, output);
  }

  template <>
  eIcicleError convert_montgomery(
    const g2_projective_t* input, uint64_t n, bool is_into, const VecOpsConfig& config, g2_projective_t* output)
  {
    return CONCAT_EXPAND(CURVE, g2_projective_convert_montgomery)(input, n, is_into, &config, output);
  }

#endif // G2
} // namespace icicle