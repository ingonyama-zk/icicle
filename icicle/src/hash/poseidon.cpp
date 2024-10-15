#include "icicle/errors.h"
#include "icicle/backend/hash/poseidon_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  ICICLE_DISPATCHER_INST(InitPoseidonConstantsDispatcher, poseidon_init_constants, InitPoseidonConstantsImpl);

  template <>
  eIcicleError poseidon_init_constants(const PoseidonConstantsOptions<scalar_t>* options)
  {
    return InitPoseidonConstantsDispatcher::execute(options);
  }

  ICICLE_DISPATCHER_INST(
    InitPoseidonDefaultConstantsDispatcher, poseidon_init_default_constants, InitPoseidonDefaultConstantsImpl);

  template <>
  eIcicleError poseidon_init_default_constants<scalar_t>()
  {
    return InitPoseidonDefaultConstantsDispatcher::execute(scalar_t::zero());
  }

  ICICLE_DISPATCHER_INST(CreatePoseidonHasherDispatcher, create_poseidon, CreatePoseidonImpl);

  template <>
  Hash create_poseidon_hash<scalar_t>(unsigned arity, unsigned default_input_size, bool is_domain_tag)
  {
    std::shared_ptr<HashBackend> backend;
    ICICLE_CHECK(
      CreatePoseidonHasherDispatcher::execute(arity, default_input_size, is_domain_tag, backend, scalar_t::zero()));
    Hash poseidon{backend};
    return poseidon;
  }

} // namespace icicle