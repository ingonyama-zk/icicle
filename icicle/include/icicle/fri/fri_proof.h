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
     * @brief Initialize the Merkle proofs and final polynomial storage for the FRI proof.
     *
     * A FriProof instance should first be created and then initialized using this method
     * before use. The initialization is done with the required parameters when setting up
     * the prover and generating the proof.
     *
     * @param nof_queries Number of queries in the proof.
     * @param nof_fri_rounds Number of FRI rounds (rounds).
     * @param final_poly_size Size of the final polynomial.
     * @return An eIcicleError indicating success or failure.
     */
    eIcicleError init(const size_t nof_queries, const size_t nof_fri_rounds, const size_t final_poly_size)
    {
      // Resize the matrix to hold nof_queries rows and nof_fri_rounds columns
      m_query_proofs.resize(
        2 * nof_queries,
        std::vector<MerkleProof>(nof_fri_rounds)); // for each query, we have 2 proofs (for the leaf and its symmetric)
      m_final_poly.resize(final_poly_size);

      return eIcicleError::SUCCESS;
    }

    /**
     * @brief Get a reference to a specific Merkle proof for a given query index in a specific FRI round. Each query
     * includes a proof for two values per round.
     *
     * This function returns a reference to a pre-allocated Merkle proof in the `m_query_proofs` array.
     * The proof is initially empty and will be populated by another function responsible for generating
     * the actual proof data.
     * @param query_idx Index of the query.
     * @param round_idx Index of the round (FRI round).
     * @return Reference to the Merkle proof at the specified position.
     */
    MerkleProof& get_query_proof_slot(const size_t query_idx, const size_t round_idx)
    {
      if (query_idx < 0 || query_idx >= m_query_proofs.size()) { throw std::out_of_range("Invalid query index"); }
      if (round_idx < 0 || round_idx >= m_query_proofs[query_idx].size()) {
        throw std::out_of_range("Invalid round index");
      }
      return m_query_proofs[query_idx][round_idx];
    }

    /**
     * @brief Get a const reference to a specific Merkle proof for a given query index in a specific FRI round. Each
     * query includes a proof for two values per round.
     */
    const MerkleProof& get_query_proof_slot(const size_t query_idx, const size_t round_idx) const
    {
      if (query_idx < 0 || query_idx >= m_query_proofs.size()) { throw std::out_of_range("Invalid query index"); }
      if (round_idx < 0 || round_idx >= m_query_proofs[query_idx].size()) {
        throw std::out_of_range("Invalid round index");
      }
      return m_query_proofs[query_idx][round_idx];
    }

    /**
     * @brief Returns a pair containing the pointer to the merkle tree root data and its size.
     * @return A pair of (root data pointer, root size).
     */
    std::pair<const std::byte*, std::size_t> get_merkle_tree_root(const size_t round_idx) const
    {
      return m_query_proofs[0 /*query_idx*/][round_idx]
        .get_root(); // Since all queries in the same round share the same root, we can just return root for query index
                     // 0 of the current round
    }

    /**
     * @brief Get a pointer to the Merkle proofs at a specific query index.
     *
     * @param query_idx The index of the query to retrieve proofs for.
     * @return Pointer to the first MerkleProof at the given query index.
     */
    const std::vector<MerkleProof>& get_proofs_at_query(const size_t query_idx) { return m_query_proofs[query_idx]; }

    /**
     * @brief Get the number of FRI queries in the proof.
     *
     * @return Number of FRI queries.
     */
    size_t get_nof_fri_queries() const { return m_query_proofs.size(); }

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
    size_t get_final_poly_size() const { return m_final_poly.size(); }

    /**
     * @brief Set the proof-of-work nonce.
     *
     * @param pow_nonce The proof-of-work nonce to set.
     */
    void set_pow_nonce(uint64_t pow_nonce) { m_pow_nonce = pow_nonce; }

    /**
     * @brief Get the proof-of-work nonce.
     *
     * @return The current proof-of-work nonce.
     */
    uint64_t get_pow_nonce() const { return m_pow_nonce; }

    /**
     * @brief Get a mutable pointer to the final polynomial data.
     *
     * @return Pointer to the first element of the final polynomial.
     */
    F* get_final_poly() { return m_final_poly.data(); }

    /**
     * @brief Get a const pointer to the final polynomial data.
     *
     * @return Const pointer to the first element of the final polynomial.
     */
    const F* get_final_poly() const { return m_final_poly.data(); }

  private:
    std::vector<std::vector<MerkleProof>>
      m_query_proofs; // Matrix of Merkle proofs [query][round] - contains path, root, leaf. for each query, we have 2
                      // proofs (for the leaf in [2*query] and its symmetric in [2*query+1])
    std::vector<F> m_final_poly; // Final polynomial (constant in canonical FRI)
    uint64_t m_pow_nonce;        // Proof-of-work nonce
  };

} // namespace icicle