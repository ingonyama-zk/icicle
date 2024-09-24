#include "icicle/errors.h"
#include "icicle/backend/hash/keccak_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  // Keccak 256
  ICICLE_DISPATCHER_INST(Keccak256Dispatcher, keccak_256_factory, KeccakFactoryImpl);

  Hash create_keccak_256_hash(uint64_t input_chunk_size)
  {
    std::shared_ptr<HashBackend> backend;
    ICICLE_CHECK(Keccak256Dispatcher::execute(input_chunk_size, backend));
    Hash keccak{backend};
    return keccak;
  }

  // Keccak 512
  ICICLE_DISPATCHER_INST(Keccak512Dispatcher, keccak_512_factory, KeccakFactoryImpl);

  Hash create_keccak_512_hash(uint64_t input_chunk_size)
  {
    std::shared_ptr<HashBackend> backend;
    ICICLE_CHECK(Keccak512Dispatcher::execute(input_chunk_size, backend));
    Hash keccak{backend};
    return keccak;
  }

  // Sha3 256
  ICICLE_DISPATCHER_INST(Sha3_256Dispatcher, sha3_256_factory, KeccakFactoryImpl);

  Hash create_sha3_256_hash(uint64_t input_chunk_size)
  {
    std::shared_ptr<HashBackend> backend;
    ICICLE_CHECK(Sha3_256Dispatcher::execute(input_chunk_size, backend));
    Hash keccak{backend};
    return keccak;
  }

  // Sha3 512
  ICICLE_DISPATCHER_INST(Sha3_512Dispatcher, sha3_512_factory, KeccakFactoryImpl);

  Hash create_sha3_512_hash(uint64_t input_chunk_size)
  {
    std::shared_ptr<HashBackend> backend;
    ICICLE_CHECK(Sha3_512Dispatcher::execute(input_chunk_size, backend));
    Hash keccak{backend};
    return keccak;
  }

} // namespace icicle