#pragma once

#include <functional>
#include "icicle/utils/utils.h"
#include "icicle/device.h"
#include "icicle/hash/blake2s.h"

namespace icicle {

  /*************************** Backend registration ***************************/
  using Blake2sFactoryImpl = std::function<eIcicleError(
    const Device& device, uint64_t input_chunk_size, std::shared_ptr<HashBackend>& backend /*OUT*/)>;

  // Blake2s 256
  void register_blake2s_factory(const std::string& deviceType, Blake2sFactoryImpl impl);

#define REGISTER_BLAKE2S_FACTORY_BACKEND(DEVICE_TYPE, FUNC)                                                            \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_blake2s) = []() -> bool {                                                                  \
      register_blake2s_factory(DEVICE_TYPE, FUNC);                                                                     \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

} // namespace icicle