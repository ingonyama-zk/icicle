#include "icicle/errors.h"
#include "icicle/fri/fri.h"
#include "icicle/backend/fri_backend.h"
#include "icicle/merkle/merkle_tree.h"
#include "icicle/dispatcher.h"
#include "icicle/hash/hash.h"
#include "icicle/utils/log.h"
#include <cstddef>

namespace icicle {

  ICICLE_DISPATCHER_INST(FriDispatcher, fri_factory, FriFactoryImpl<scalar_t>);

  /**
   * @brief Create a FRI instance.
   * @return A `Fri<scalar_t>` object built around the chosen backend.
   */
  template <typename S>
  Fri<S> create_fri_with_merkle_trees(
    size_t folding_factor,
    size_t stopping_degree,
    std::vector<MerkleTree> merkle_trees)
  {
    std::shared_ptr<FriBackend<S>> backend; 
    ICICLE_CHECK(FriDispatcher::execute(
      folding_factor,
      stopping_degree,
      merkle_trees,
      backend));

    Fri<S> fri{backend};
    return fri;
  }

  /**
  * @brief Specialization of create_fri for the case of 
  *        (input_size, folding_factor, stopping_degree, hash_for_merkle_tree, output_store_min_layer).
  */
  template <>
  Fri<scalar_t> create_fri(
    size_t input_size,
    size_t folding_factor,
    size_t stopping_degree,
    Hash& hash_for_merkle_tree,
    uint64_t output_store_min_layer)
  {
    ICICLE_ASSERT(folding_factor == 2) << "Only folding factor of 2 is supported";
    size_t log_input_size = static_cast<size_t>(std::log2(static_cast<double>(input_size)));
    size_t df = stopping_degree;
    size_t log_df_plus_1 = (df > 0) ? static_cast<size_t>(std::log2(static_cast<double>(df + 1))) : 0;
    size_t fold_rounds = (log_input_size > log_df_plus_1) ? (log_input_size - log_df_plus_1) : 0;

    std::vector<MerkleTree> merkle_trees;
    merkle_trees.reserve(fold_rounds);
    std::vector<Hash> hashes_for_merkle_tree_vec(fold_rounds, hash_for_merkle_tree);
    for (size_t i = 0; i < fold_rounds; i++) {
      merkle_trees.emplace_back(MerkleTree::create(hashes_for_merkle_tree_vec, sizeof(scalar_t), output_store_min_layer));
      hashes_for_merkle_tree_vec.pop_back();
    }
    return create_fri_with_merkle_trees<scalar_t>(
      folding_factor,
      stopping_degree,
      merkle_trees);
  }

  /**
  * @brief Specialization of create_fri for the case of 
  *        (folding_factor, stopping_degree, vector<MerkleTree>&&).
  */
  template <>
  Fri<scalar_t> create_fri(
    size_t folding_factor,
    size_t stopping_degree,
    std::vector<MerkleTree> merkle_trees)
  {
    return create_fri_with_merkle_trees<scalar_t>(
      folding_factor,
      stopping_degree,
      merkle_trees);
  }

} // namespace icicle
