#include "icicle/errors.h"
#include "icicle/fri/fri.h"
#include "icicle/backend/fri_backend.h"
#include "icicle/dispatcher.h"
#include <cstddef>

namespace icicle {

  ICICLE_DISPATCHER_INST(FriDispatcher, fri_factory, FriFactoryImpl<scalar_t>);


  /**
  * @brief Specialization of create_fri for the case of 
  *        (input_size, folding_factor, stopping_degree, hash_for_merkle_tree, output_store_min_layer).
  */
  template <>
  Fri<scalar_t> create_fri<scalar_t>(
    size_t input_size,
    size_t folding_factor,
    size_t stopping_degree,
    const Hash& hash_for_merkle_tree,
    uint64_t output_store_min_layer)
  {
    std::shared_ptr<FriBackend<scalar_t>> backend;
    ICICLE_CHECK(FriDispatcher::execute(
      input_size,
      folding_factor,
      stopping_degree,
      hash_for_merkle_tree,
      output_store_min_layer,
      backend));

    Fri<scalar_t> fri{backend};
    return fri;
  }

  /**
  * @brief Specialization of create_fri for the case of 
  *        (folding_factor, stopping_degree, vector<MerkleTree>&&).
  */
  template <>
  Fri<scalar_t> create_fri<scalar_t>(
    size_t folding_factor,
    size_t stopping_degree,
    std::vector<MerkleTree>&& merkle_trees)
  {
    std::shared_ptr<FriBackend<scalar_t>> backend; 
    ICICLE_CHECK(FriDispatcher::execute(
      folding_factor,
      stopping_degree,
      std::move(merkle_trees),
      backend));

    Fri<scalar_t> fri{backend};
    return fri;
  }

} // namespace icicle
