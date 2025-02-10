#include "icicle/errors.h"
#include "icicle/backend/hash/pow_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  ICICLE_DISPATCHER_INST(PowSolverDispatcher, pow_solver, PowSolverImpl);

  extern "C" eIcicleError pow(
    Hash& hasher,
    uint8_t* challenge,
    uint32_t challenge_size,
    uint32_t padding_size,
    uint8_t bits,
    const PowConfig& config,
    bool* found,
    uint64_t* nonce,
    uint64_t* mined_hash)
  {
    return PowSolverDispatcher::execute(
      hasher, challenge, challenge_size, padding_size, bits, config, found, nonce, mined_hash);
  }
} // namespace icicle