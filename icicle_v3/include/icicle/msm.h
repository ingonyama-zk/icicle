#pragma once

#include <functional>

#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/utils.h"
#include "icicle/config_extension.h"

#include "icicle/curves/affine.h"
#include "icicle/curves/projective.h"
#include "icicle/fields/field.h"
#include "icicle/curves/curve_config.h"

using namespace curve_config;

namespace icicle {

  /*************************** Frontend APIs ***************************/
  struct MSMConfig {
    icicleStreamHandle stream; /**< stream for async execution. */
    int nof_bases; /**< Number of bases in the MSM for batched MSM. Set to 0 if all MSMs use the same bases or set to
                    * 'batch X #scalars' otherwise.  Default value: 0 (that is reuse bases for all batch elements). */
    int precompute_factor;            /**< The number of extra points to pre-compute for each point. See the
                                       *   [precompute_msm_bases](@ref precompute_msm_bases) function, `precompute_factor` passed
                                       *   there needs to be equal to the one used here. Larger values decrease the
                                       *   number of computations to make, on-line memory footprint, but increase the static
                                       *   memory footprint. Default value: 1 (i.e. don't pre-compute). */
    int batch_size;                   /**< The number of MSMs to compute. Default value: 1. */
    bool are_scalars_on_device;       /**< True if scalars are on device and false if they're on host. Default value:
                                       *   false. */
    bool are_scalars_montgomery_form; /**< True if scalars are in Montgomery form and false otherwise. Default value:
                                       *   true. */
    bool are_points_on_device; /**< True if points are on device and false if they're on host. Default value: false. */
    bool are_points_montgomery_form; /**< True if coordinates of points are in Montgomery form and false otherwise.
                                      *   Default value: true. */
    bool are_results_on_device; /**< True if the results should be on device and false if they should be on host. If set
                                 *   to false, `is_async` won't take effect because a synchronization is needed to
                                 *   transfer results to the host. Default value: false. */
    bool is_async;              /**< Whether to run the MSM asynchronously. If set to true, the MSM function will be
                                 *   non-blocking and you'd need to synchronize it explicitly by running
                                 *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the MSM
                                 *   function will block the current CPU thread. */

    ConfigExtension* ext = nullptr; /** backend specific extensions*/
  };

  /**
   * A function that returns the default value of [MSMConfig](@ref MSMConfig) for the [MSM](@ref MSM) function.
   * @return Default value of [MSMConfig](@ref MSMConfig).
   */
  static MSMConfig default_msm_config()
  {
    MSMConfig config = {
      nullptr, // stream
      0,       // nof_bases
      1,       // precompute_factor
      1,       // batch_size
      false,   // are_scalars_on_device
      false,   // are_scalars_montgomery_form
      false,   // are_points_on_device
      false,   // are_points_montgomery_form
      false,   // are_results_on_device
      false,   // is_async
    };
    return config;
  }

  /**
   * A function that computes MSM: \f$ MSM(s_i, P_i) = \sum_{i=1}^N s_i \cdot P_i \f$.
   * @param scalars Scalars \f$ s_i \f$. In case of batch MSM, the scalars from all MSMs are concatenated.
   * @param bases Bases \f$ P_i \f$. In case of batch MSM, all *unique* points are concatenated.
   * So, if for example all MSMs share the same base points, they can be repeated only once.
   * @param msm_size MSM size \f$ N \f$. If a batch of MSMs (which all need to have the same size) is computed, this is
   * the size of 1 MSM.
   * @param config [MSMConfig](@ref MSMConfig) used in this MSM.
   * @param results Buffer for the result (or results in the case of batch MSM).
   * @tparam S Scalar field type.
   * @tparam A The type of points \f$ \{P_i\} \f$ which is typically an [affine
   * Weierstrass](https://hyperelliptic.org/EFD/g1p/auto-shortw.html) point.
   * @tparam P Output type, which is typically a [projective
   * Weierstrass](https://hyperelliptic.org/EFD/g1p/auto-shortw-projective.html) point in our codebase.
   * @return `SUCCESS` if the execution was successful and an error code otherwise.
   *
   */
  template <typename S, typename A, typename P>
  eIcicleError msm(const S* scalars, const A* bases, int msm_size, const MSMConfig& config, P* results);

  struct MsmPreComputeConfig {
    icicleStreamHandle stream; /**< stream for async execution. */
    bool is_input_on_device;
    bool is_output_on_device;
    bool is_async;

    ConfigExtension* ext = nullptr; /** backend specific extensions*/
  };

  static MsmPreComputeConfig default_msm_pre_compute_config()
  {
    MsmPreComputeConfig config = {
      nullptr, // stream
      false,   // is_input_on_device
      false,   // is_output_on_device
      false,   // is_async
    };
    return config;
  }

  template <typename A>
  eIcicleError msm_precompute_bases(
    const A* input_bases, int nof_bases, int precompute_factor, const MsmPreComputeConfig& config, A* output_bases);

  /*************************** Backend registration ***************************/

  using MsmImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* scalars,
    const affine_t* bases,
    int msm_size,
    const MSMConfig& config,
    projective_t* results)>;

  void register_msm(const std::string& deviceType, MsmImpl impl);

#define REGISTER_MSM_BACKEND(DEVICE_TYPE, FUNC)                                                                        \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_msm) = []() -> bool {                                                                      \
      register_msm(DEVICE_TYPE, FUNC);                                                                                 \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using MsmPreComputeImpl = std::function<eIcicleError(
    const Device& device,
    const affine_t* input_bases,
    int nof_bases,
    int precompute_factor,
    const MsmPreComputeConfig& config,
    affine_t* output_bases)>;

  void register_msm_precompute_bases(const std::string& deviceType, MsmPreComputeImpl impl);

#define REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND(DEVICE_TYPE, FUNC)                                                      \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_msm_precompute_bases) = []() -> bool {                                                     \
      register_msm_precompute_bases(DEVICE_TYPE, FUNC);                                                                \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

#ifdef G2
  using MsmG2Impl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* scalars,
    const g2_affine_t* bases,
    int msm_size,
    const MSMConfig& config,
    g2_projective_t* results)>;

  void register_msm_g2(const std::string& deviceType, MsmG2Impl impl);

#define REGISTER_MSM_G2_BACKEND(DEVICE_TYPE, FUNC)                                                                     \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_msm_g2) = []() -> bool {                                                                   \
      register_msm_g2(DEVICE_TYPE, FUNC);                                                                              \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using MsmG2PreComputeImpl = std::function<eIcicleError(
    const Device& device,
    const g2_affine_t* input_bases,
    int nof_bases,
    int precompute_factor,
    const MsmPreComputeConfig& config,
    g2_affine_t* output_bases)>;

  void register_msm_g2_precompute_bases(const std::string& deviceType, MsmG2PreComputeImpl impl);

#define REGISTER_MSM_G2_PRE_COMPUTE_BASES_BACKEND(DEVICE_TYPE, FUNC)                                                   \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_msm_g2_precompute_bases) = []() -> bool {                                                  \
      register_msm_g2_precompute_bases(DEVICE_TYPE, FUNC);                                                             \
      return true;                                                                                                     \
    }();                                                                                                               \
  }
#endif // G2

}; // namespace icicle