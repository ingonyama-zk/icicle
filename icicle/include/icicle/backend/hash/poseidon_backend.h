#pragma once

#include <functional>
#include "icicle/utils/utils.h"
#include "icicle/device.h"
#include "icicle/hash/poseidon.h"

#include "icicle/fields/field_config.h"
using namespace field_config;

namespace icicle {

  /*************************** Backend registration ***************************/

  using InitPoseidonConstantsImpl = std::function<eIcicleError(
    const Device& device,
    unsigned arity,
    unsigned alpha,
    unsigned full_rounds_half,
    unsigned partial_rounds,
    const scalar_t* rounds_constants,
    const scalar_t* mds_matrix,
    const scalar_t* non_sparse_matrix,
    const scalar_t* sparse_matrices,
    const scalar_t* domain_tag,
    std::shared_ptr<PoseidonConstants<scalar_t>>& constants /*out*/)>;

  // poseidon init constants
  void register_poseidon_init_constants(const std::string& deviceType, InitPoseidonConstantsImpl impl);

#define REGISTER_POSEIDON_INIT_CONSTANTS_BACKEND(DEVICE_TYPE, FUNC)                                                    \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_poseidon_init_constants) = []() -> bool {                                                  \
      register_poseidon_init_constants(DEVICE_TYPE, FUNC);                                                             \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using InitPoseidonDefaultConstantsImpl = std::function<eIcicleError(
    const Device& device, unsigned arity, std::shared_ptr<PoseidonConstants<scalar_t>>& constants /*out*/)>;

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
    const Device& device, std::shared_ptr<PoseidonConstants<scalar_t>>, std::shared_ptr<HashBackend>& /*OUT*/)>;

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