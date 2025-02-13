#pragma once

#include <functional>
#include "icicle/utils/utils.h"
#include "icicle/device.h"
#include "icicle/hash/blake3.h"

namespace icicle {

  /*************************** Backend registration ***************************/
  using Blake3FactoryImpl = std::function<eIcicleError(
    const Device& device, uint64_t input_chunk_size, std::shared_ptr<HashBackend>& backend /*OUT*/)>;

  // Blake3 256
  void register_blake3_factory(const std::string& deviceType, Blake3FactoryImpl impl);

#define REGISTER_BLAKE3_FACTORY_BACKEND(DEVICE_TYPE, FUNC)                                                             \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_blake3) = []() -> bool {                                                                   \
      register_blake3_factory(DEVICE_TYPE, FUNC);                                                                      \
      return true;                                                                                                     \
    }();                                                                                                               \
  }
} // namespace icicle