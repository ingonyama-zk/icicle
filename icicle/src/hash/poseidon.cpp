#include "icicle/errors.h"
#include "icicle/backend/hash/poseidon_backend.h"
#include "icicle/dispatcher.h"

namespace icicle {

  ICICLE_DISPATCHER_INST(InitPoseidonConstantsDispatcher, poseidon_init_constants, InitPoseidonConstantsImpl);

  template <>
  eIcicleError poseidon_init_constants(
    unsigned arity,
    unsigned alpha,
    unsigned full_rounds_half,
    unsigned partial_rounds,
    const scalar_t* rounds_constants,
    const scalar_t* mds_matrix,
    const scalar_t* non_sparse_matrix,
    const scalar_t* sparse_matrices,
    const scalar_t* domain_tag,
    std::shared_ptr<PoseidonConstants<scalar_t>>& constants /*out*/)
  {
    return InitPoseidonConstantsDispatcher::execute(
      arity, alpha, full_rounds_half, partial_rounds, rounds_constants, mds_matrix, non_sparse_matrix, sparse_matrices,
      domain_tag, constants);
  }

  ICICLE_DISPATCHER_INST(
    InitPoseidonDefaultConstantsDispatcher, poseidon_init_default_constants, InitPoseidonDefaultConstantsImpl);

  template <>
  eIcicleError
  poseidon_init_default_constants(unsigned arity, std::shared_ptr<PoseidonConstants<scalar_t>>& constants /*out*/)
  {
    return InitPoseidonDefaultConstantsDispatcher::execute(arity, constants);
  }

  ICICLE_DISPATCHER_INST(CreatePoseidonHasherDispatcher, create_poseidon, CreatePoseidonImpl);

  template <>
  Hash create_poseidon_hash(std::shared_ptr<PoseidonConstants<scalar_t>> constants)
  {
    std::shared_ptr<HashBackend> backend;
    ICICLE_CHECK(CreatePoseidonHasherDispatcher::execute(constants, backend));
    Hash poseidon{backend};
    return poseidon;
  }

} // namespace icicle