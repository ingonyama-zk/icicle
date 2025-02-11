#pragma once

#include <stddef.h>
#include <stdint.h>
#include <functional>
#include "icicle/utils/utils.h"
#include "icicle/hash/pow.h"

namespace icicle {

  using PowSolverImpl = std::function<eIcicleError(
    const Device& device,
    Hash& hasher,
    std::byte* challenge,
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

  using PowCheckImpl = std::function<eIcicleError(
    const Device& device,
    Hash& hasher,
    std::byte* challenge,
    uint32_t challenge_size,
    uint8_t solution_bits,
    const PowConfig& config,
    uint64_t nonce,
    bool& is_correct,
    uint64_t& mined_hash)>;

  void register_pow_check(const std::string& deviceType, PowCheckImpl impl);

#define REGISTER_POW_CHECK_BACKEND(DEVICE_TYPE, FUNC)                                                                  \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_pow) = []() -> bool {                                                                      \
      register_pow_check(DEVICE_TYPE, FUNC);                                                                           \
      return true;                                                                                                     \
    }();                                                                                                               \
  }
} // namespace icicle