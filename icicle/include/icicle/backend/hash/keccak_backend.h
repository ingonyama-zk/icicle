#pragma once

#include <functional>
#include "icicle/utils/utils.h"
#include "icicle/device.h"
#include "icicle/hash/keccak.h"

namespace icicle {

  /*************************** Backend registration ***************************/
  using KeccakFactoryImpl = std::function<eIcicleError(
    const Device& device, uint64_t input_chunk_size, std::shared_ptr<HashBackend>& backend /*OUT*/)>;

  // Keccak 256
  void register_keccak_256_factory(const std::string& deviceType, KeccakFactoryImpl impl);

#define REGISTER_KECCAK_256_FACTORY_BACKEND(DEVICE_TYPE, FUNC)                                                         \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_keccak256) = []() -> bool {                                                                \
      register_keccak_256_factory(DEVICE_TYPE, FUNC);                                                                  \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  // Keccak 512
  void register_keccak_512_factory(const std::string& deviceType, KeccakFactoryImpl impl);

#define REGISTER_KECCAK_512_FACTORY_BACKEND(DEVICE_TYPE, FUNC)                                                         \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_keccak512) = []() -> bool {                                                                \
      register_keccak_512_factory(DEVICE_TYPE, FUNC);                                                                  \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  // SHA3 256
  void register_sha3_256_factory(const std::string& deviceType, KeccakFactoryImpl impl);

#define REGISTER_SHA3_256_FACTORY_BACKEND(DEVICE_TYPE, FUNC)                                                           \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_sha3_256) = []() -> bool {                                                                 \
      register_sha3_256_factory(DEVICE_TYPE, FUNC);                                                                    \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  // SHA3 512
  void register_sha3_512_factory(const std::string& deviceType, KeccakFactoryImpl impl);

#define REGISTER_SHA3_512_FACTORY_BACKEND(DEVICE_TYPE, FUNC)                                                           \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_sha3_512) = []() -> bool {                                                                 \
      register_sha3_512_factory(DEVICE_TYPE, FUNC);                                                                    \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

} // namespace icicle