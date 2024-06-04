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

  struct ConvertMontgomeryConfig {
    icicleStreamHandle stream; /**< stream for async execution. */
    bool is_input_on_device;
    bool is_output_on_device;
    bool is_async;

    ConfigExtension ext; /** backend specific extensions*/
  };

  static ConvertMontgomeryConfig default_convert_montgomery_config()
  {
    ConvertMontgomeryConfig config = {
      nullptr, // stream
      false,   // is_input_on_device
      false,   // is_output_on_device
      false,   // is_async
    };
    return config;
  }

  template <typename T>
  eIcicleError
  points_convert_montgomery(const T* input, size_t n, bool is_into, const ConvertMontgomeryConfig& config, T* output);

  /*************************** Backend registration ***************************/

  using AffineConvertMontImpl = std::function<eIcicleError(
    const Device& device,
    const affine_t* input,
    size_t n,
    bool is_into,
    const ConvertMontgomeryConfig& config,
    affine_t* output)>;

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
    const ConvertMontgomeryConfig& config,
    projective_t* output)>;

  void register_projective_convert_montgomery(const std::string& deviceType, ProjectiveConvertMontImpl);

#define REGISTER_PROJECTIVE_CONVERT_MONTGOMERY_BACKEND(DEVICE_TYPE, FUNC)                                              \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_projective_convert_mont) = []() -> bool {                                                  \
      register_projective_convert_montgomery(DEVICE_TYPE, FUNC);                                                       \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

}; // namespace icicle
