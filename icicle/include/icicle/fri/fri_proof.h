#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include "icicle/backend/merkle/merkle_tree_backend.h"
#include "icicle/merkle/merkle_tree.h"
#include "icicle/utils/log.h"

namespace icicle {

  /**
   * @brief Represents a FRI proof.
   *
   * @tparam F Type of the field element (e.g., prime field or extension field elements).
   */

  template <typename F>
  class FriProof
  {
  public:
    // Constructor
    FriProof() : m_pow_nonce(0) {}

    /**
     * @brief Initialize the Merkle proofs and final polynomial storage.
     *
     * @param nof_queries Number of queries in the proof.
     * @param nof_fri_rounds Number of FRI rounds (rounds).
     */
    void init(const size_t nof_queries, const size_t nof_fri_rounds, const size_t final_poly_size)
    {
      ICICLE_ASSERT(nof_queries > 0 && nof_fri_rounds > 0)
        << "Number of queries and FRI rounds must be > 0. nof_queries = " << nof_queries
        << ", nof_fri_rounds = " << nof_fri_rounds;

      // Resize the matrix to hold nof_queries rows and nof_fri_rounds columns
      m_query_proofs.resize(
        2 * nof_queries,
        std::vector<MerkleProof>(nof_fri_rounds)); // for each query, we have 2 proofs (for the leaf and its symmetric)
      m_final_poly_size = final_poly_size;
      m_final_poly = std::make_unique<F[]>(final_poly_size);
    }

    /**
     * @brief Get a reference to a specific Merkle proof.
     *
     * @param query_idx Index of the query.
     * @param round_idx Index of the round (FRI round).
     * @return Reference to the Merkle proof at the specified position.
     */
    MerkleProof& get_query_proof(const size_t query_idx, const size_t round_idx)
    {
      if (query_idx < 0 || query_idx >= m_query_proofs.size()) { throw std::out_of_range("Invalid query index"); }
      if (round_idx < 0 || round_idx >= m_query_proofs[query_idx].size()) {
        throw std::out_of_range("Invalid round index");
      }
      return m_query_proofs[query_idx][round_idx];
    }

    /**
     * @brief Returns a pair containing the pointer to the root data and its size.
     * @return A pair of (root data pointer, root size).
     */
    std::pair<const std::byte*, std::size_t> get_root(const size_t round_idx) const
    {
      return m_query_proofs[0][round_idx].get_root();
    }

    // /**
    //  * @brief Returns a tuple containing the pointer to the leaf data, its size and index.
    //  * @return A tuple of (leaf data pointer, leaf size, leaf_index).
    //  */
    // std::tuple<const std::byte*, std::size_t, uint64_t> get_leaf(const size_t query_idx, const size_t round_idx)
    // const
    // {
    //   return m_query_proofs[query_idx][round_idx].get_leaf();
    // }

    /**
     * @brief Get the number of FRI rounds in the proof.
     *
     * @return Number of FRI rounds.
     */
    size_t get_nof_fri_rounds() const { return m_query_proofs[0].size(); }

    /**
     * @brief Get the final poly size.
     *
     * @return final_poly_size.
     */
    size_t get_final_poly_size() const { return m_final_poly_size; }

    void set_pow_nonce(uint64_t pow_nonce) { m_pow_nonce = pow_nonce; }

    uint64_t get_pow_nonce() const { return m_pow_nonce; }

    // get pointer to the final polynomial
    F* get_final_poly() const { return m_final_poly.get(); }

  private:
    std::vector<std::vector<MerkleProof>>
      m_query_proofs; // Matrix of Merkle proofs [query][round] - contains path, root, leaf. for each query, we have 2
                      // proofs (for the leaf in [2*query] and its symmetric in [2*query+1])
    std::unique_ptr<F[]> m_final_poly; // Final polynomial (constant in canonical FRI)
    size_t m_final_poly_size;          // Size of the final polynomial
    uint64_t m_pow_nonce;              // Proof-of-work nonce
  };

} // namespace icicle