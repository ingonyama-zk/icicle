#pragma once

#include <functional>

#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/utils.h"
#include "icicle/config_extension.h"

#include "icicle/curves/affine.h"
#include "icicle/curves/projective.h"
#include "icicle/vec_ops.h"
#include "icicle/fields/field.h"
#include "icicle/curves/curve_config.h"

using namespace curve_config;

namespace icicle {

  /*************************** Backend registration ***************************/

  using AffineConvertMontImpl = std::function<eIcicleError(
    const Device& device, const affine_t* input, size_t n, bool is_into, const VecOpsConfig& config, affine_t* output)>;

  void register_affine_convert_montgomery(const std::string& deviceType, AffineConvertMontImpl);

#define REGISTER_AFFINE_CONVERT_MONTGOMERY_BACKEND(DEVICE_TYPE, FUNC)                                                  \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_affine_convert_mont) = []() -> bool {                                                      \
      register_affine_convert_montgomery(DEVICE_TYPE, FUNC);                                                           \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using ProjectiveConvertMontImpl = std::function<eIcicleError(
    const Device& device,
    const projective_t* input,
    size_t n,
    bool is_into,
    const VecOpsConfig& config,
    projective_t* output)>;

  void register_projective_convert_montgomery(const std::string& deviceType, ProjectiveConvertMontImpl);

#define REGISTER_PROJECTIVE_CONVERT_MONTGOMERY_BACKEND(DEVICE_TYPE, FUNC)                                              \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_projective_convert_mont) = []() -> bool {                                                  \
      register_projective_convert_montgomery(DEVICE_TYPE, FUNC);                                                       \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

#ifdef G2
  using AffineG2ConvertMontImpl = std::function<eIcicleError(
    const Device& device,
    const g2_affine_t* input,
    size_t n,
    bool is_into,
    const VecOpsConfig& config,
    g2_affine_t* output)>;

  void register_g2_affine_convert_montgomery(const std::string& deviceType, AffineG2ConvertMontImpl);

  #define REGISTER_AFFINE_G2_CONVERT_MONTGOMERY_BACKEND(DEVICE_TYPE, FUNC)                                             \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_affine_g2_convert_mont) = []() -> bool {                                                 \
        register_g2_affine_convert_montgomery(DEVICE_TYPE, FUNC);                                                      \
        return true;                                                                                                   \
      }();                                                                                                             \
    }

  using ProjectiveG2ConvertMontImpl = std::function<eIcicleError(
    const Device& device,
    const g2_projective_t* input,
    size_t n,
    bool is_into,
    const VecOpsConfig& config,
    g2_projective_t* output)>;

  void register_g2_projective_convert_montgomery(const std::string& deviceType, ProjectiveG2ConvertMontImpl);

  #define REGISTER_PROJECTIVE_G2_CONVERT_MONTGOMERY_BACKEND(DEVICE_TYPE, FUNC)                                         \
    namespace {                                                                                                        \
      static bool UNIQUE(_reg_projective_g2_convert_mont) = []() -> bool {                                             \
        register_g2_projective_convert_montgomery(DEVICE_TYPE, FUNC);                                                  \
        return true;                                                                                                   \
      }();                                                                                                             \
    }
#endif // G2

}; // namespace icicle
