#include <cuda.h>
#include <stdexcept>

#include "icicle/errors.h"
#include "icicle/curves/montgomery_conversion.h"
#include "gpu-utils/error_handler.h"
#include "error_translation.h"
#include "cuda_mont.cuh"

#include "icicle/curves/curve_config.h"
using namespace curve_config;
using namespace icicle;

namespace icicle {

  template <typename T>
  eIcicleError
  cuda_convert_mont(const Device& device, const T* input, size_t n, bool is_into, const VecOpsConfig& config, T* output)
  {
    cudaError_t err = is_into ? montgomery::ConvertMontgomery<T, true>(input, n, config, output)
                              : montgomery::ConvertMontgomery<T, false>(input, n, config, output);
    return translateCudaError(err);
  }

  REGISTER_AFFINE_CONVERT_MONTGOMERY_BACKEND("CUDA", cuda_convert_mont<affine_t>);
  REGISTER_PROJECTIVE_CONVERT_MONTGOMERY_BACKEND("CUDA", cuda_convert_mont<projective_t>);

#ifdef G2
  REGISTER_AFFINE_G2_CONVERT_MONTGOMERY_BACKEND("CUDA", cuda_convert_mont<g2_affine_t>);
  REGISTER_PROJECTIVE_G2_CONVERT_MONTGOMERY_BACKEND("CUDA", cuda_convert_mont<g2_projective_t>);
#endif // G2

} // namespace icicle