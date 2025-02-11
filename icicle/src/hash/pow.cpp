#include "icicle/errors.h"
#include "icicle/backend/hash/pow_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  ICICLE_DISPATCHER_INST(PowSolverDispatcher, pow_solver, PowSolverImpl);

  extern "C" eIcicleError proof_of_work(
    Hash& hasher,
    std::byte* challenge,
    uint32_t challenge_size,
    uint8_t solution_bits,
    const PowConfig& config,
    bool& found,
    uint64_t& nonce,
    uint64_t& mined_hash)
  {
    return PowSolverDispatcher::execute(
      hasher, challenge, challenge_size, solution_bits, config, found, nonce, mined_hash);
  }

  ICICLE_DISPATCHER_INST(PowCheckDispatcher, pow_check, PowCheckImpl);

  extern "C" eIcicleError proof_of_work_check(
    Hash& hasher,
    std::byte* challenge,
    uint32_t challenge_size,
    uint8_t solution_bits,
    const PowConfig& config,
    uint64_t nonce,
    bool& is_correct,
    uint64_t& mined_hash)
  {
    return PowCheckDispatcher::execute(
      hasher, challenge, challenge_size, solution_bits, config, nonce, is_correct, mined_hash);
  }
} // namespace icicle