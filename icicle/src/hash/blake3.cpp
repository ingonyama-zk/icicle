#include "icicle/errors.h"
#include "icicle/backend/hash/blake3_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  // Blake3
  ICICLE_DISPATCHER_INST(Blake3Dispatcher, blake3_factory, Blake3FactoryImpl);

  Hash create_blake3_hash(uint64_t input_chunk_size)
  {
    std::shared_ptr<HashBackend> backend;
    ICICLE_CHECK(Blake3Dispatcher::execute(input_chunk_size, backend));
    Hash blake3{backend};
    return blake3;
  }

} // namespace icicle