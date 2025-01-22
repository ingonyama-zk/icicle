#pragma once

#include "icicle/pow.h"

namespace icicle {

  using PowBlake3Impl = std::function<eIcicleError(
    const Device& device, 
    uint8_t* challenge, 
    uint8_t bits, 
    const PowConfig& config, 
    bool* found, 
    uint64_t* nonce, 
    uint64_t* mined_hash)>;

  void register_pow_blake3(const std::string& deviceType, PowBlake3Impl impl);

#define REGISTER_POW_BACKEND(DEVICE_TYPE, FUNC)                                                                        \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_pow) = []() -> bool {                                                                      \
      register_pow_blake3(DEVICE_TYPE, FUNC);                                                                                 \
      return true;                                                                                                     \
    }();                                                                                                               \
  }
}