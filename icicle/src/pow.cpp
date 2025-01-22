#include "icicle/errors.h"
#include "icicle/backend/hash/blake3_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  ICICLE_DISPATCHER_INST(PowBlake3Dispatcher, pow_blake3, PowBlake3Impl);

  extern "C" eIcicleError some_pow_blake3(uint8_t* challenge, uint8_t bits, const PowConfig& config, bool* found, uint64_t* nonce, uint64_t* mined_hash) {
    return PowBlake3Dispatcher::execute(challenge, bits, config, found, nonce, mined_hash);
  }
} // namespace icicle