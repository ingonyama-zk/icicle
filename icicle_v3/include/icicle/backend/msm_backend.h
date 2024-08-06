#pragma once

#include "icicle/msm.h"
#include "icicle/curves/curve_config.h"

using namespace curve_config;

namespace icicle {
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
    int bases_size,
    const MSMConfig& config,
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

  void register_g2_msm(const std::string& deviceType, MsmG2Impl impl);

#define REGISTER_MSM_G2_BACKEND(DEVICE_TYPE, FUNC)                                                                     \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_msm_g2) = []() -> bool {                                                                   \
      register_g2_msm(DEVICE_TYPE, FUNC);                                                                              \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using MsmG2PreComputeImpl = std::function<eIcicleError(
    const Device& device,
    const g2_affine_t* input_bases,
    int bases_size,
    const MSMConfig& config,
    g2_affine_t* output_bases)>;

  void register_g2_msm_precompute_bases(const std::string& deviceType, MsmG2PreComputeImpl impl);

#define REGISTER_MSM_G2_PRE_COMPUTE_BASES_BACKEND(DEVICE_TYPE, FUNC)                                                   \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_msm_g2_precompute_bases) = []() -> bool {                                                  \
      register_g2_msm_precompute_bases(DEVICE_TYPE, FUNC);                                                             \
      return true;                                                                                                     \
    }();                                                                                                               \
  }
#endif // G2
} // namespace icicle
