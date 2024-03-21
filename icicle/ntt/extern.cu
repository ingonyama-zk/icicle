#include "kernel_ntt.cu"
#include "ntt.cu"

#include "gpu-utils/device_context.cuh"
#include "utils/utils.h"

namespace ntt {
  // Explicit instantiation for scalar type
  template cudaError_t generate_external_twiddles_generic(
    const scalar_t& basic_root,
    scalar_t* external_twiddles,
    scalar_t*& internal_twiddles,
    scalar_t*& basic_twiddles,
    uint32_t log_size,
    cudaStream_t& stream);

  template cudaError_t generate_external_twiddles_fast_twiddles_mode(
    const scalar_t& basic_root,
    scalar_t* external_twiddles,
    scalar_t*& internal_twiddles,
    scalar_t*& basic_twiddles,
    uint32_t log_size,
    cudaStream_t& stream);

  template cudaError_t mixed_radix_ntt<scalar_t, scalar_t>(
    scalar_t* d_input,
    scalar_t* d_output,
    scalar_t* external_twiddles,
    scalar_t* internal_twiddles,
    scalar_t* basic_twiddles,
    int ntt_size,
    int max_logn,
    int batch_size,
    bool columns_batch,
    bool is_inverse,
    bool fast_tw,
    Ordering ordering,
    scalar_t* arbitrary_coset,
    int coset_gen_index,
    cudaStream_t cuda_stream);

  /**
   * Extern "C" version of [InitDomain](@ref InitDomain) function with the following
   * value of template parameter (where the curve is given by `-DFIELD` env variable during build):
   *  - `S` is the [scalar field](@ref scalar_t) of the curve;
   */
  extern "C" cudaError_t CONCAT_EXPAND(FIELD, InitializeDomain)(
    scalar_t* primitive_root, device_context::DeviceContext& ctx, bool fast_twiddles_mode)
  {
    return InitDomain(*primitive_root, ctx, fast_twiddles_mode);
  }

  /**
   * Extern "C" version of [NTT](@ref NTT) function with the following values of template parameters
   * (where the curve is given by `-DFIELD` env variable during build):
   *  - `S` and `E` are both the [scalar field](@ref scalar_t) of the curve;
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t
  CONCAT_EXPAND(FIELD, NTTCuda)(scalar_t* input, int size, NTTDir dir, NTTConfig<scalar_t>& config, scalar_t* output)
  {
    return NTT<scalar_t, scalar_t>(input, size, dir, config, output);
  }

#if defined(ECNTT_DEFINED)
  /**
   * Extern "C" version of [NTT](@ref NTT) function with the following values of template parameters
   * (where the curve is given by `-DFIELD` env variable during build):
   *  - `S` is the [projective representation](@ref projective_t) of the curve (i.e. EC NTT is computed);
   *  - `E` is the [scalar field](@ref scalar_t) of the curve;
   * @return `cudaSuccess` if the execution was successful and an error code otherwise.
   */
  extern "C" cudaError_t CONCAT_EXPAND(CURVE, ECNTTCuda)(
    projective_t* input, int size, NTTDir dir, NTTConfig<scalar_t>& config, projective_t* output)
  {
    return NTT<scalar_t, projective_t>(input, size, dir, config, output);
  }
#endif
} // namespace ntt