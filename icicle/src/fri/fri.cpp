#include "icicle/fri/fri.h"
#include "icicle/dispatcher.h"

namespace icicle {

  using FriFactoryScalar = FriFactoryImpl<scalar_t, scalar_t>;
  ICICLE_DISPATCHER_INST(FriDispatcher, fri_factory, FriFactoryScalar);
  /**
   * @brief Create a FRI instance.
   * @return A `Fri<scalar_t>` object built around the chosen backend.
   */
  template <typename S, typename F>
  Fri<S, F> create_fri_with_merkle_trees(
    const size_t folding_factor, const size_t stopping_degree, std::vector<MerkleTree> merkle_trees)
  {
    std::shared_ptr<FriBackend<S, F>> backend;
    ICICLE_CHECK(FriDispatcher::execute(folding_factor, stopping_degree, std::move(merkle_trees), backend));

    Fri<S, F> fri{backend};
    return fri;
  }

  /**
   * @brief Specialization of create_fri for the case of
   *        (input_size, folding_factor, stopping_degree, hash_for_merkle_tree, output_store_min_layer).
   */
  template <typename S, typename F>
  Fri<S, F> create_fri_template(
    const size_t input_size,
    const size_t folding_factor,
    const size_t stopping_degree,
    Hash merkle_tree_leaves_hash,
    Hash merkle_tree_compress_hash,
    const uint64_t output_store_min_layer)
  {
    const size_t log_input_size = static_cast<size_t>(std::log2(static_cast<double>(input_size)));
    const size_t df = stopping_degree;
    const size_t log_df_plus_1 = (df > 0) ? static_cast<size_t>(std::log2(static_cast<double>(df + 1))) : 0;
    const size_t fold_rounds = (log_input_size > log_df_plus_1) ? (log_input_size - log_df_plus_1) : 0;

    std::vector<MerkleTree> merkle_trees;
    merkle_trees.reserve(fold_rounds);
    size_t compress_hash_arity =
      merkle_tree_compress_hash.default_input_chunk_size() / merkle_tree_compress_hash.output_size();
    if (compress_hash_arity != 2) { ICICLE_LOG_ERROR << "Currently only compress hash arity of 2 is supported"; }
    size_t first_merkle_tree_height = std::ceil(std::log2(input_size) / std::log2(compress_hash_arity)) + 1;
    std::vector<Hash> layer_hashes(first_merkle_tree_height, merkle_tree_compress_hash);
    layer_hashes[0] = merkle_tree_leaves_hash;
    uint64_t leaf_element_size = merkle_tree_leaves_hash.default_input_chunk_size();
    for (size_t i = 0; i < fold_rounds; i++) {
      merkle_trees.emplace_back(MerkleTree::create(layer_hashes, leaf_element_size, output_store_min_layer));
      layer_hashes.pop_back();
    }
    return create_fri_with_merkle_trees<S, F>(folding_factor, stopping_degree, std::move(merkle_trees));
  }

#ifdef EXT_FIELD
  using FriExtFactoryScalar = FriFactoryImpl<scalar_t, extension_t>;
  ICICLE_DISPATCHER_INST(FriExtFieldDispatcher, extension_fri_factory, FriExtFactoryScalar);
  /**
   * @brief Create a FRI instance.
   * @return A `Fri<scalar_t>` object built around the chosen backend.
   */
  template <typename S, typename F>
  Fri<S, F> create_fri_with_merkle_trees_ext(
    const size_t folding_factor, const size_t stopping_degree, std::vector<MerkleTree> merkle_trees)
  {
    std::shared_ptr<FriBackend<S, F>> backend;
    ICICLE_CHECK(FriExtFieldDispatcher::execute(folding_factor, stopping_degree, std::move(merkle_trees), backend));

    Fri<S, F> fri{backend};
    return fri;
  }
  /**
   * @brief Specialization of create_fri for the case of
   *        (input_size, folding_factor, stopping_degree, hash_for_merkle_tree, output_store_min_layer).
   */
  template <typename S, typename F>
  Fri<S, F> create_fri_template_ext(
    const size_t input_size,
    const size_t folding_factor,
    const size_t stopping_degree,
    Hash merkle_tree_leaves_hash,
    Hash merkle_tree_compress_hash,
    const uint64_t output_store_min_layer)
  {
    const size_t log_input_size = static_cast<size_t>(std::log2(static_cast<double>(input_size)));
    const size_t df = stopping_degree;
    const size_t log_df_plus_1 = (df > 0) ? static_cast<size_t>(std::log2(static_cast<double>(df + 1))) : 0;
    const size_t fold_rounds = (log_input_size > log_df_plus_1) ? (log_input_size - log_df_plus_1) : 0;

    std::vector<MerkleTree> merkle_trees;
    merkle_trees.reserve(fold_rounds);
    size_t compress_hash_arity =
      merkle_tree_compress_hash.default_input_chunk_size() / merkle_tree_compress_hash.output_size();
    if (compress_hash_arity != 2) { ICICLE_LOG_ERROR << "Currently only compress hash arity of 2 is supported"; }
    size_t first_merkle_tree_height = std::ceil(std::log2(input_size) / std::log2(compress_hash_arity)) + 1;
    std::vector<Hash> layer_hashes(first_merkle_tree_height, merkle_tree_compress_hash);
    layer_hashes[0] = merkle_tree_leaves_hash;
    uint64_t leaf_element_size = merkle_tree_leaves_hash.default_input_chunk_size();
    for (size_t i = 0; i < fold_rounds; i++) {
      merkle_trees.emplace_back(MerkleTree::create(layer_hashes, leaf_element_size, output_store_min_layer));
      layer_hashes.pop_back();
    }
    return create_fri_with_merkle_trees_ext<S, F>(folding_factor, stopping_degree, std::move(merkle_trees));
  }

#endif // EXT_FIELD

  template <>
  Fri<scalar_t, scalar_t> create_fri(
    const size_t input_size,
    const size_t folding_factor,
    const size_t stopping_degree,
    Hash merkle_tree_leaves_hash,
    Hash merkle_tree_compress_hash,
    const uint64_t output_store_min_layer)
  {
    return create_fri_template<scalar_t, scalar_t>(
      input_size, folding_factor, stopping_degree, merkle_tree_leaves_hash, merkle_tree_compress_hash,
      output_store_min_layer);
  }

#ifdef EXT_FIELD
  template <>
  Fri<scalar_t, extension_t> create_fri(
    const size_t input_size,
    const size_t folding_factor,
    const size_t stopping_degree,
    Hash merkle_tree_leaves_hash,
    Hash merkle_tree_compress_hash,
    const uint64_t output_store_min_layer)
  {
    return create_fri_template_ext<scalar_t, extension_t>(
      input_size, folding_factor, stopping_degree, merkle_tree_leaves_hash, merkle_tree_compress_hash,
      output_store_min_layer);
  }
#endif

} // namespace icicle
