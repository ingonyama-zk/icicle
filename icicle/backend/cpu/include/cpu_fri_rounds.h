namespace icicle {

  template <typename F>
  class FriRounds
  {
  public:
    /**
     * @brief Constructor that accepts an already-existing array of Merkle trees.
     *        Ownership is transferred from the caller.
     *
     * @param merkle_trees A vector of MerkleTrees.
     * @param log_input_size The log of the input size.
     */
    FriRounds(std::vector<MerkleTree>& merkle_trees, const size_t log_input_size) : m_merkle_trees(merkle_trees)
    {
      size_t fold_rounds = m_merkle_trees.size();
      m_rounds_evals.resize(fold_rounds);
      for (size_t i = 0; i < fold_rounds; i++) {
        m_rounds_evals[i] = std::make_unique<F[]>(1ULL << (log_input_size - i));
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
      if (round_idx >= m_merkle_trees.size()) {
        ICICLE_LOG_ERROR << "round index out of bounds";
        return nullptr;
      }
      return &m_merkle_trees[round_idx];
    }

    F* get_round_evals(size_t round_idx)
    {
      if (round_idx >= m_merkle_trees.size()) {
        ICICLE_LOG_ERROR << "round index out of bounds";
        return nullptr;
      }
      return m_rounds_evals[round_idx].get();
    }

    /**
     * @brief Retrieve the Merkle root for a specific fri round.
     *
     * @param round_idx The index of the round.
     * @return A pair containing a pointer to the Merkle root bytes and its size.
     */
    std::pair<const std::byte*, size_t> get_merkle_root_for_round(size_t round_idx) const
    {
      if (round_idx >= m_merkle_trees.size()) { return {nullptr, 0}; }
      return m_merkle_trees[round_idx].get_merkle_root();
    }

  private:
    // Persistent polynomial evaluations for each round (heap allocated).
    // For round i, the expected length is 2^(m_initial_log_size - i).
    std::vector<std::unique_ptr<F[]>> m_rounds_evals;

    // Holds MerkleTree for each round. m_merkle_trees[i] is the tree for round i.
    std::vector<MerkleTree> m_merkle_trees;
  };

} // namespace icicle
