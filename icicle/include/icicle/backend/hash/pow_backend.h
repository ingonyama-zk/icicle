#pragma once

#include <stddef.h>
#include <stdint.h>
#include <functional>
#include "icicle/utils/utils.h"
#include "icicle/hash/pow.h"

namespace icicle {

  using PowSolverImpl = std::function<eIcicleError(
    const Device& device,
    const Hash& hasher,
    const std::byte* challenge,
    uint32_t challenge_size,
    uint8_t solution_bits,
    const PowConfig& config,
    bool& found,
    uint64_t& nonce,
    uint64_t& mined_hash)>;

  void register_pow_solver(const std::string& deviceType, PowSolverImpl impl);

#define REGISTER_POW_SOLVER_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_pow) = []() -> bool {                                                                      \
      register_pow_solver(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using PowVerifyImpl = std::function<eIcicleError(
    const Device& device,
    const Hash& hasher,
    const std::byte* challenge,
    uint32_t challenge_size,
    uint8_t solution_bits,
    const PowConfig& config,
    uint64_t nonce,
    bool& is_correct,
    uint64_t& mined_hash)>;

  void register_pow_verify(const std::string& deviceType, PowVerifyImpl impl);

#define REGISTER_POW_VERIFY_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_pow) = []() -> bool {                                                                      \
      register_pow_verify(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }
} // namespace icicle