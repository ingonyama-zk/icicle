#pragma once

#include <functional>

#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/utils.h"
#include "icicle/config_extension.h"

#include "icicle/curves/affine.h"
#include "icicle/curves/projective.h"
#include "icicle/fields/field.h"

namespace icicle {

  /*************************** Frontend APIs ***************************/
  /**
   * @struct MSMConfig
   * @brief Configuration for Multi-Scalar Multiplication (MSM).
   */
  struct MSMConfig {
    icicleStreamHandle stream; /**< Stream for asynchronous execution. */
    int precompute_factor;     /**< Number of extra points to pre-compute for each point. See the
                                *   precompute_msm_bases function; precompute_factor passed there needs to be equal to the
                                * one used here. Larger values decrease the number of computations to make, on-line memory
                                * footprint, but increase the static memory footprint. Default value: 1 (i.e., don't
                                * pre-compute). */
    int c; /**< \f$ c \f$ value, or "window bitsize", which is the main parameter of the "bucket method" used to solve
            * the MSM problem. Larger value means more on-line memory footprint but also more parallelism and less
            * computational complexity (up to a certain point). Default value: 0 (the optimal value of \f$ c \f$ is
            * chosen automatically). */
    int bitsize;                /**< Number of bits of the largest scalar. Typically equals the bitsize of scalar field,
                                 *   but if a different (better) upper bound is known, it should be reflected in this variable.
                                 *   Default value: 0 (set to the bitsize of scalar field). */
    int batch_size;             /**< Number of MSMs to compute. Default value: 1. */
    bool are_bases_shared;      /**< Bases are shared for batch. Set to true if all MSMs use the same bases. Otherwise, the number 
                                    of bases and number of scalars are expected to be equal. Default value: true. */
    bool are_scalars_on_device; /**< True if scalars are on device, false if they're on host. Default value: false. */
    bool
      are_scalars_montgomery_form; /**< True if scalars are in Montgomery form, false otherwise. Default value: true. */
    bool are_points_on_device;     /**< True if points are on device, false if they're on host. Default value: false. */
    bool are_points_montgomery_form; /**< True if coordinates of points are in Montgomery form, false otherwise. Default
                                        value: true. */
    bool are_results_on_device; /**< True if the results should be on device, false if they should be on host. If set
                                 *   to false, is_async won't take effect because a synchronization is needed to
                                 *   transfer results to the host. Default value: false. */
    bool is_async;              /**< Whether to run the MSM asynchronously. If set to true, the MSM function will be
                                 *   non-blocking and you'd need to synchronize it explicitly by running
                                 *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the MSM
                                 *   function will block the current CPU thread. */
    ConfigExtension* ext = nullptr; /**< Backend-specific extensions. */
  };

  /**
   * @brief Returns the default value of MSMConfig for the MSM function.
   *
   * @return Default value of MSMConfig.
   */
  static MSMConfig default_msm_config()
  {
    MSMConfig config = {
      nullptr, // stream
      1,       // precompute_factor
      0,       // c
      0,       // bitsize
      1,       // batch_size
      true,    // are_bases_shared
      false,   // are_scalars_on_device
      false,   // are_scalars_montgomery_form
      false,   // are_points_on_device
      false,   // are_points_montgomery_form
      false,   // are_results_on_device
      false,   // is_async
      nullptr, // ext
    };
    return config;
  }

  /**
   * @brief Computes MSM: \f$ MSM(s_i, P_i) = \sum_{i=1}^N s_i \cdot P_i \f$.
   *
   * @tparam S Scalar field type.
   * @tparam A Type of bases \f$ \{P_i\} \f$ (typically an affine Weierstrass point).
   * @tparam P Output type (typically a projective Weierstrass point).
   * @param scalars Scalars \f$ s_i \f$. In case of batch MSM, the scalars from all MSMs are concatenated.
   * @param bases Bases \f$ P_i \f$. In case of batch MSM, all unique points are concatenated, or shared.
   * @param msm_size MSM size \f$ N \f$. If a batch of MSMs is computed, this is the size of one MSM.
   * @param config Configuration for the MSM operation.
   * @param results Buffer for the result (or results in the case of batch MSM).
   * @return `SUCCESS` if the execution was successful, and an error code otherwise.
   */
  template <typename S, typename A, typename P>
  eIcicleError msm(const S* scalars, const A* bases, int msm_size, const MSMConfig& config, P* results);

  /**
   * @brief Precomputes bases for MSM.
   *
   * @tparam A Type of points (typically an affine Weierstrass point).
   * @param input_bases Input bases for MSM precomputation.
   * @param bases_size Number of input bases.
   * @param config Configuration for the MSM precomputation.
   * @param output_bases Buffer to store the precomputed bases.
   * @return `SUCCESS` if the execution was successful, and an error code otherwise.
   */
  template <typename A>
  eIcicleError msm_precompute_bases(const A* input_bases, int bases_size, const MSMConfig& config, A* output_bases);

}; // namespace icicle