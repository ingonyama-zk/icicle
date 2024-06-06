
#include "icicle/curves/montgomery_conversion.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"

#include "icicle/curves/curve_config.h"

using namespace curve_config;
using namespace icicle;

template <typename T>
eIcicleError cpu_convert_mont(
  const Device& device, const T* input, size_t n, bool is_into, const ConvertMontgomeryConfig& config, T* output)
{
  for (size_t i = 0; i < n; ++i) {
    output[i] = is_into ? T::to_montgomery(input[i]) : T::from_montgomery(input[i]);
  }
  return eIcicleError::SUCCESS;
}

REGISTER_AFFINE_CONVERT_MONTGOMERY_BACKEND("CPU", cpu_convert_mont<affine_t>);
REGISTER_PROJECTIVE_CONVERT_MONTGOMERY_BACKEND("CPU", cpu_convert_mont<projective_t>);

#ifdef G2
REGISTER_AFFINE_G2_CONVERT_MONTGOMERY_BACKEND("CPU", cpu_convert_mont<g2_affine_t>);
REGISTER_PROJECTIVE_G2_CONVERT_MONTGOMERY_BACKEND("CPU", cpu_convert_mont<g2_projective_t>);
#endif // G2