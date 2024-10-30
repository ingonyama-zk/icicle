#pragma once

#include <functional>
#include "icicle/utils/utils.h"
#include "icicle/device.h"
#include "icicle/hash/poseidon.h"

#include "icicle/fields/field_config.h"
using namespace field_config;

namespace icicle {

  /*************************** Backend registration ***************************/

  using InitPoseidonConstantsImpl =
    std::function<eIcicleError(const Device& device, const PoseidonConstantsOptions<scalar_t>* options)>;

  // poseidon init constants
  void register_poseidon_init_constants(const std::string& deviceType, InitPoseidonConstantsImpl impl);

#define REGISTER_POSEIDON_INIT_CONSTANTS_BACKEND(DEVICE_TYPE, FUNC)                                                    \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_poseidon_init_constants) = []() -> bool {                                                  \
      register_poseidon_init_constants(DEVICE_TYPE, FUNC);                                                             \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  // Note: 'phantom' is a workaround for the function required per field but need to differentiate by type when
  // calling.
  using InitPoseidonDefaultConstantsImpl = std::function<eIcicleError(const Device& device, const scalar_t& phantom)>;

  // poseidon init constants
  void register_poseidon_init_default_constants(const std::string& deviceType, InitPoseidonDefaultConstantsImpl impl);

#define REGISTER_POSEIDON_INIT_DEFAULT_CONSTANTS_BACKEND(DEVICE_TYPE, FUNC)                                            \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_poseidon_init_default_constants) = []() -> bool {                                          \
      register_poseidon_init_default_constants(DEVICE_TYPE, FUNC);                                                     \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using CreatePoseidonImpl = std::function<eIcicleError(
    const Device& device,
    unsigned arity,
    unsigned default_input_size,
    bool is_domain_tag,
    scalar_t domain_tag_value,
    bool use_all_zeroes_padding,
    std::shared_ptr<HashBackend>& /*OUT*/,
    const scalar_t& phantom)>;

  // poseidon init constants
  void register_create_poseidon(const std::string& deviceType, CreatePoseidonImpl impl);

#define REGISTER_CREATE_POSEIDON_BACKEND(DEVICE_TYPE, FUNC)                                                            \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_create_poseidon) = []() -> bool {                                                          \
      register_create_poseidon(DEVICE_TYPE, FUNC);                                                                     \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

} // namespace icicle