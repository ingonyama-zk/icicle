#pragma once

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
    FriProof() : m_pow_nonce(0){}

     /**
     * @brief Initialize the Merkle proofs and final polynomial storage.
     *
     * @param nof_queries Number of queries in the proof.
     * @param nof_fri_rounds Number of FRI rounds (rounds).
     */
    void init(const size_t nof_queries, const size_t nof_fri_rounds, const size_t m_stopping_degree)
    {
      ICICLE_ASSERT(nof_queries > 0 && nof_fri_rounds > 0)
          << "Number of queries and FRI rounds must be > 0. nof_queries = " << nof_queries << ", nof_fri_rounds = " << nof_fri_rounds;
      
      // Resize the matrix to hold nof_queries rows and nof_fri_rounds columns
      m_query_proofs.resize(2*nof_queries, std::vector<MerkleProof>(nof_fri_rounds)); //for each query, we have 2 proofs (for the leaf and its symmetric)
      m_final_poly = std::make_unique<F[]>(m_stopping_degree + 1);
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
      if (query_idx < 0 || query_idx >= m_query_proofs.size()) {
        throw std::out_of_range("Invalid query index");
      }
      if (round_idx < 0 || round_idx >= m_query_proofs[query_idx].size()) {
        throw std::out_of_range("Invalid round index");
      }
      return m_query_proofs[query_idx][round_idx];
    }

    void set_pow_nonce(uint64_t pow_nonce)
    {
      m_pow_nonce = pow_nonce;
    }

    uint64_t get_pow_nonce() const
    {
      return m_pow_nonce;
    }

    //get pointer to the final polynomial
    F* get_final_poly() const
    {
      return m_final_poly.get();
    }

  private:
    std::vector<std::vector<MerkleProof>> m_query_proofs; // Matrix of Merkle proofs [query][round] - contains path, root, leaf
    std::unique_ptr<F[]> m_final_poly;                    // Final polynomial (constant in canonical FRI)
    uint64_t m_pow_nonce;                                 // Proof-of-work nonce

  public:
    // for debug
    void print_proof()
    {
      std::cout << "FRI Proof:" << std::endl;
      for (int query_idx = 0; query_idx < m_query_proofs.size(); query_idx++) {
        std::cout << " Query " << query_idx << ":" << std::endl;
        for (int round_idx = 0; round_idx < m_query_proofs[query_idx].size(); round_idx++) {
          std::cout << "  round " << round_idx << ":" << std::endl;
          m_query_proofs[query_idx][round_idx].print_proof();
        }
      }
    }
  };

} // namespace icicle