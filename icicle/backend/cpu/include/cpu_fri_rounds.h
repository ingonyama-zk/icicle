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
    * @brief Constructor that accepts an already-existing array of Merkle trees.
    *        Ownership is transferred from the caller.
    *
    * @param merkle_trees A moved vector of `unique_ptr<MerkleTree>`.
    */
  FriRounds(std::vector<MerkleTree>&& merkle_trees)
    : m_merkle_trees(std::move(merkle_trees))
  {
    size_t fold_rounds = m_merkle_trees.size(); //FIXME - consider stopping degree?
    m_round_evals.resize(fold_rounds);
    for (size_t i = 0; i < fold_rounds; i++) {
      m_round_evals[i] = std::make_unique<F[]>(1ULL << (fold_rounds - i));
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
    return &m_merkle_trees[round_idx];
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
    return m_merkle_trees[round_idx].get_merkle_root();
  }

private:
  // Persistent polynomial evaluations for each round (heap allocated).
  // For round i, the expected length is 2^(m_initial_log_size - i).
  std::vector<std::unique_ptr<F[]>> m_round_evals;

  // Holds MerkleTree for each round. m_merkle_trees[i] is the tree for round i.
  std::vector<MerkleTree> m_merkle_trees;
};

} // namespace icicle
