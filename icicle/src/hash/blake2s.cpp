#include "icicle/errors.h"
#include "icicle/backend/hash/blake2s_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  // Blake2s
  ICICLE_DISPATCHER_INST(Blake2sDispatcher, blake2s_factory, Blake2sFactoryImpl);

  Hash create_blake2s_hash(uint64_t total_input_limbs)
  {
    std::shared_ptr<HashBackend> backend;
    ICICLE_CHECK(Blake2sDispatcher::execute(total_input_limbs, backend));
    Hash keccak{backend};
    return keccak;
  }


  /*************************** C API ***************************/

  extern "C" Hash* create_blake2s_hash_c_api(uint64_t total_input_limbs)
  {
    return new Hash(create_blake2s_hash(total_input_limbs));
  }


  // TODO Yuval : need to expose one deleter from C++. This will be used to drop any object

} // namespace icicle