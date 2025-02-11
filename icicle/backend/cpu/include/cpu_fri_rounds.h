#pragma once

#include <cstddef>
#include <vector>
#include <memory>
#include "icicle/merkle/merkle_tree.h"
#include "icicle/errors.h"
#include "icicle/hash/hash.h"
#include "icicle/hash/keccak.h"
#include "icicle/utils/log.h"


namespace icicle {

template <typename F>
class FriRounds
{
public:
  /**
    * @brief Constructor that stores parameters for building Merkle trees.
    *
    * @param folding_factor         The factor by which the codeword is folded each round.
    * @param stopping_degree        The polynomial degree threshold to stop folding.
    * @param hash_for_merkle_tree   Hash function used for each Merkle tree layer.
    * @param output_store_min_layer Minimum layer index to store fully in the output (default=0).
    */
  FriRounds(size_t log_input_size,
            size_t folding_factor,
            size_t stopping_degree,
            const Hash& hash_for_merkle_tree,
            uint64_t output_store_min_layer = 0)
  {
    ICICLE_ASSERT(folding_factor == 2) << "Only folding factor of 2 is supported";
    size_t df = stopping_degree;
    size_t log_df_plus_1 = (df > 0) ? static_cast<size_t>(std::log2(static_cast<double>(df + 1))) : 0;
    size_t fold_rounds = (log_input_size > log_df_plus_1) ? (log_input_size - log_df_plus_1) : 0;

    m_round_evals.resize(fold_rounds);
    m_merkle_trees.reserve(fold_rounds);
    std::vector<Hash> hashes_for_merkle_tree_vec(fold_rounds, hash_for_merkle_tree);
    for (size_t i = 0; i < fold_rounds; i++) {
      m_merkle_trees.push_back(std::make_unique<MerkleTree>(hashes_for_merkle_tree_vec, sizeof(F), output_store_min_layer));
      hashes_for_merkle_tree_vec.pop_back();
      m_round_evals[i] = std::make_unique<std::vector<F>>();
      m_round_evals[i]->reserve(1ULL << (log_input_size - i));
    }
  }

  /**
    * @brief Constructor that accepts an already-existing array of Merkle trees.
    *        Ownership is transferred from the caller.
    *
    * @param merkle_trees A moved vector of `unique_ptr<MerkleTree>`.
    */
  FriRounds(std::vector<std::unique_ptr<MerkleTree>>&& merkle_trees)
    : m_merkle_trees(std::move(merkle_trees))
  {
    size_t fold_rounds = m_merkle_trees.size();
    m_round_evals.resize(fold_rounds);
    for (size_t i = 0; i < fold_rounds; i++) {
      m_round_evals[i] = std::make_unique<std::vector<F>>();
      m_round_evals[i]->reserve(1ULL << (fold_rounds - i));
    }
  }

  /**
  * @brief Get the Merkle tree for a specific fri round.
  *
  * @param round_idx The index of the fri round.
  * @return A pointer to the Merkle tree backend for the specified fri round.
  */
  MerkleTree* get_merkle_tree(size_t round_idx)
  {
    ICICLE_ASSERT(round_idx < m_merkle_trees.size()) << "round index out of bounds";
    return m_merkle_trees[round_idx].get();
  }

  F* get_round_evals(size_t round_idx)
  {
    ICICLE_ASSERT(round_idx < m_round_evals.size()) << "round index out of bounds";
    return m_round_evals[round_idx].get();
  }

  /**
    * @brief Retrieve the Merkle root for a specific fri round.
    *
    * @param round_idx The index of the round.
    * @return A pair containing a pointer to the Merkle root bytes and its size.
    */
  std::pair<const std::byte*, size_t> get_merkle_root_for_round(size_t round_idx) const
  {
    if (round_idx >= m_merkle_trees.size()) {
      return {nullptr, 0};
    }
    return m_merkle_trees[round_idx]->get_merkle_root();
  }

private:
  // Persistent polynomial evaluations for each round (heap allocated).
  // For round i, the expected length is 2^(m_initial_log_size - i).
  std::vector<std::unique_ptr<F>> m_round_evals;

  // Holds unique ownership of each MerkleTree for each round. m_merkle_trees[i] is the tree for round i.
  std::vector<std::unique_ptr<MerkleTree>> m_merkle_trees;
};

} // namespace icicle
