#include "icicle/errors.h"
#include "icicle/backend/hash/keccak_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  /*************************** Keccak ***************************/

  // Keccak 256
  ICICLE_DISPATCHER_INST(Keccak256Dispatcher, keccak_256_factory, KeccakFactoryImpl);

  // TODO Yuval expose C API

  Hash create_keccak_256_hash(uint64_t total_input_limbs)
  {
    std::shared_ptr<HashBackend> backend;

    // TODO Yuval: what to do with the return value??
    auto err = Keccak256Dispatcher::execute(total_input_limbs, backend);
    Hash keccak{backend};
    return keccak;
  }

  // Keccak 512
  ICICLE_DISPATCHER_INST(Keccak512Dispatcher, keccak_512_factory, KeccakFactoryImpl);

  // TODO Yuval expose C API

  Hash create_keccak_512_hash(uint64_t total_input_limbs)
  {
    std::shared_ptr<HashBackend> backend;

    // TODO Yuval: what to do with the return value??
    auto err = Keccak512Dispatcher::execute(total_input_limbs, backend);
    Hash keccak{backend};
    return keccak;
  }

  // Sha3 256
  ICICLE_DISPATCHER_INST(Sah3_256Dispatcher, sha3_256_factory, KeccakFactoryImpl);

  // TODO Yuval expose C API

  Hash create_sha3_256_hash(uint64_t total_input_limbs)
  {
    std::shared_ptr<HashBackend> backend;

    // TODO Yuval: what to do with the return value??
    auto err = Keccak256Dispatcher::execute(total_input_limbs, backend);
    Hash keccak{backend};
    return keccak;
  }

  // Keccak 512
  ICICLE_DISPATCHER_INST(Sah3_512Dispatcher, sha3_512_factory, KeccakFactoryImpl);

  // TODO Yuval expose C API

  Hash create_sha3_512_hash(uint64_t total_input_limbs)
  {
    std::shared_ptr<HashBackend> backend;

    // TODO Yuval: what to do with the return value??
    auto err = Keccak512Dispatcher::execute(total_input_limbs, backend);
    Hash keccak{backend};
    return keccak;
  }

} // namespace icicle