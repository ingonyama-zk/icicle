#pragma once

#include "icicle/ntt.h"
#include "icicle/fields/field_config.h"

using namespace field_config;

namespace icicle {

  /*************************** Backend registration ***************************/

  /*************************** NTT ***************************/
  using NttImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* input,
    int size,
    NTTDir dir,
    const NTTConfig<scalar_t>& config,
    scalar_t* output)>;

  void register_ntt(const std::string& deviceType, NttImpl impl);

#define REGISTER_NTT_BACKEND(DEVICE_TYPE, FUNC)                                                                        \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_ntt) = []() -> bool {                                                                      \
      register_ntt(DEVICE_TYPE, FUNC);                                                                                 \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

#ifdef EXT_FIELD
  using NttExtFieldImpl = std::function<eIcicleError(
    const Device& device,
    const extension_t* input,
    int size,
    NTTDir dir,
    const NTTConfig<scalar_t>& config,
    extension_t* output)>;

  void register_extension_ntt(const std::string& deviceType, NttExtFieldImpl impl);

#define REGISTER_NTT_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                              \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_ntt_ext_field) = []() -> bool {                                                            \
      register_extension_ntt(DEVICE_TYPE, FUNC);                                                                       \
      return true;                                                                                                     \
    }();                                                                                                               \
  }
#endif // EXT_FIELD

  /*************************** INIT DOMAIN ***************************/
  using NttInitDomainImpl = std::function<eIcicleError(
    const Device& device, const scalar_t& primitive_root, const NTTInitDomainConfig& config)>;

  void register_ntt_init_domain(const std::string& deviceType, NttInitDomainImpl);

#define REGISTER_NTT_INIT_DOMAIN_BACKEND(DEVICE_TYPE, FUNC)                                                            \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_ntt_init_domain) = []() -> bool {                                                          \
      register_ntt_init_domain(DEVICE_TYPE, FUNC);                                                                     \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  /*************************** RELEASE DOMAIN ***************************/
  // Note: 'dummy' is a workaround for the function required per field but need to differentiate by type when
  // calling. TODO Yuval: avoid this param somehow
  using NttReleaseDomainImpl = std::function<eIcicleError(const Device& device, const scalar_t& dummy)>;

  void register_ntt_release_domain(const std::string& deviceType, NttReleaseDomainImpl);

#define REGISTER_NTT_RELEASE_DOMAIN_BACKEND(DEVICE_TYPE, FUNC)                                                         \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_ntt_release_domain) = []() -> bool {                                                       \
      register_ntt_release_domain(DEVICE_TYPE, FUNC);                                                                  \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  /*************************** GET ROU FROM DOMAIN ***************************/
  using NttGetRouFromDomainImpl = std::function<eIcicleError(const Device& device, uint64_t logn, scalar_t* rou)>;

  void register_ntt_get_rou_from_domain(const std::string& deviceType, NttGetRouFromDomainImpl);

#define REGISTER_NTT_GET_ROU_FROM_DOMAIN_BACKEND(DEVICE_TYPE, FUNC)                                                    \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_ntt_get_rou_from_domain) = []() -> bool {                                                  \
      register_ntt_get_rou_from_domain(DEVICE_TYPE, FUNC);                                                             \
      return true;                                                                                                     \
    }();                                                                                                               \
  }
} // namespace icicle