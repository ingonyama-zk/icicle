#include "icicle/errors.h"
#include "icicle/backend/hash/poseidon_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  ICICLE_DISPATCHER_INST(CreatePoseidonHasherDispatcher, create_poseidon, CreatePoseidonImpl);

  template <>
  Hash create_poseidon_hash<scalar_t>(unsigned t, bool use_domain_tag)
  {
    std::shared_ptr<HashBackend> backend;
    ICICLE_CHECK(CreatePoseidonHasherDispatcher::execute(t, use_domain_tag, backend, scalar_t::zero()));
    Hash poseidon{backend};
    return poseidon;
  }

} // namespace icicle