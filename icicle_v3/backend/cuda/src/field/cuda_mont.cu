#include <cuda.h>
#include <stdexcept>

#include "icicle/errors.h"
#include "icicle/vec_ops.h"
#include "gpu-utils/error_handler.h"
#include "error_translation.h"
#include "cuda_mont.cuh"

namespace icicle {

#include "icicle/fields/field_config.h"
  using namespace field_config;

  template <typename F>
  eIcicleError convert_montgomery_cuda(
    const Device& device, const F* input, uint64_t n, bool is_into, const VecOpsConfig& config, F* output)
  {
    auto err = is_into ? montgomery::ConvertMontgomery<F, true>(input, n, config, output)
                       : montgomery::ConvertMontgomery<F, false>(input, n, config, output);
    return translateCudaError(err);
  }

  REGISTER_CONVERT_MONTGOMERY_BACKEND("CUDA", convert_montgomery_cuda<scalar_t>);

#ifdef EXT_FIELD
  REGISTER_CONVERT_MONTGOMERY_EXT_FIELD_BACKEND("CUDA", convert_montgomery_cuda<extension_t>);
#endif // EXT_FIELD

} // namespace icicle
