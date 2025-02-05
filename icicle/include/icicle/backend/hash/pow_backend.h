#pragma once

#include <stddef.h>
#include <stdint.h>
#include <functional>
#include "icicle/utils/utils.h"
#include "icicle/hash/pow.h"

namespace icicle {

  using PowSolverImpl = std::function<eIcicleError(
    const Device& device, 
    Hash hasher, 
    uint8_t* challenge, 
    uint32_t challenge_size, 
    uint32_t padding_size, 
    uint8_t bits, 
    const PowConfig& config, 
    bool* found, 
    uint64_t* nonce, 
    uint64_t* mined_hash)>;

  void register_pow_solver(const std::string& deviceType, PowSolverImpl impl);

#define REGISTER_POW_SOLVER_BACKEND(DEVICE_TYPE, FUNC)                                                                        \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_pow) = []() -> bool {                                                                      \
      register_pow_solver(DEVICE_TYPE, FUNC);                                                                                 \
      return true;                                                                                                     \
    }();                                                                                                               \
  }
} // namespace icicle