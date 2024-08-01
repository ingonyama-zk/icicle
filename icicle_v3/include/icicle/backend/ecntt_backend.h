#pragma once

#include <functional>

#include "icicle/errors.h"
#include "icicle/runtime.h"

#include "icicle/ntt.h"
#include "icicle/curves/curve_config.h"
#include "icicle/utils/utils.h"

using namespace curve_config;

namespace icicle {
  /*************************** Backend registration***************************/
  using ECNttFieldImpl = std::function<eIcicleError(
    const Device& device,
    const projective_t* input,
    int size,
    NTTDir dir,
    const NTTConfig<scalar_t>& config,
    projective_t* output)>;

  void register_ecntt(const std::string& deviceType, ECNttFieldImpl impl);

#define REGISTER_ECNTT_BACKEND(DEVICE_TYPE, FUNC)                                                                      \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_ecntt_ext_field) = []() -> bool {                                                          \
      register_ecntt(DEVICE_TYPE, FUNC);                                                                               \
      return true;                                                                                                     \
    }();                                                                                                               \
  }
} // namespace icicle