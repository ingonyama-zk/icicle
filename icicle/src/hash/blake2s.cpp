#include "icicle/errors.h"
#include "icicle/backend/hash/blake2s_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  // Blake2s
  ICICLE_DISPATCHER_INST(Blake2sDispatcher, blake2s_factory, Blake2sFactoryImpl);

  Hash create_blake2s_hash(uint64_t input_chunk_size)
  {
    std::shared_ptr<HashBackend> backend;
    ICICLE_CHECK(Blake2sDispatcher::execute(input_chunk_size, backend));
    Hash blake2s{backend};
    return blake2s;
  }
} // namespace icicle