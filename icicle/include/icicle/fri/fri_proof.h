#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include "icicle/backend/merkle/merkle_tree_backend.h"
#include "icicle/merkle/merkle_tree.h"
#include "icicle/utils/log.h"
#include "icicle/serialization.h"
#include <typeinfo>

namespace icicle {

  /**
   * @brief Represents a FRI proof.
   *
   * @tparam F Type of the field element (e.g., prime field or extension field elements).
   */

  template <typename F>
  class FriProof: public Serializer
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

    void set_pow_nonce(uint64_t pow_nonce) { m_pow_nonce = pow_nonce; }

    uint64_t get_pow_nonce() const { return m_pow_nonce; }

    // get pointer to the final polynomial
    F* get_final_poly() { return m_final_poly.data(); }
    const F* get_final_poly() const { return m_final_poly.data(); }

    eIcicleError serialized_size(size_t& size) const override
    {
      size = 0;
      size += sizeof(size_t); // nof_queries
      for (const auto& query_proofs : m_query_proofs) {
        size += sizeof(size_t); // nof_fri_rounds
        for (const auto& proof : query_proofs) {
          size_t proof_size = 0;
          eIcicleError err = proof.serialized_size(proof_size);
          if (err != eIcicleError::SUCCESS) {
            return err;
          }
          size += proof_size;
        }
      }
      size += sizeof(size_t); // F type
      size += sizeof(size_t); // final_poly_size
      size += m_final_poly.size() * sizeof(F);
      size += sizeof(uint64_t); // pow_nonce

      return eIcicleError::SUCCESS;
    }

    eIcicleError serialize(std::byte*& out) const override
    {
      size_t m_query_proofs_size = m_query_proofs.size();
      std::memcpy(out, &m_query_proofs_size, sizeof(size_t));
      out += sizeof(size_t);
      for (const std::vector<MerkleProof>& query_proofs : m_query_proofs) {
        size_t query_proofs_size = query_proofs.size();
        std::memcpy(out, &query_proofs_size, sizeof(size_t));
        out += sizeof(size_t);
        for (const MerkleProof& proof : query_proofs) {
          eIcicleError err = proof.serialize(out);
          if (err != eIcicleError::SUCCESS) {
            return err;
          }
        }
      }
      size_t F_type = typeid(F).hash_code();
      std::memcpy(out, &F_type, sizeof(size_t));
      out += sizeof(size_t);
      size_t final_poly_size = m_final_poly.size();
      std::memcpy(out, &final_poly_size, sizeof(size_t));
      out += sizeof(size_t);
      std::memcpy(out, m_final_poly.data(), final_poly_size * sizeof(F));
      out += final_poly_size * sizeof(F);
      std::memcpy(out, &m_pow_nonce, sizeof(uint64_t));
      out += sizeof(uint64_t);
      return eIcicleError::SUCCESS;
    }

    eIcicleError deserialize(std::byte*& in, size_t& length) override
    {
      auto advance = [&](size_t bytes) -> bool {
        if (length < bytes) return false;
        in += bytes;
        length -= bytes;
        return true;
      };

      size_t required_length = sizeof(size_t) + sizeof(size_t) + sizeof(size_t) + sizeof(uint64_t);
      if (length < required_length) {
        ICICLE_LOG_ERROR << "Deserialization failed: length < required_length";
        return eIcicleError::COPY_FAILED;
      }
      size_t nof_queries;
      std::memcpy(&nof_queries, in, sizeof(size_t));
      advance(sizeof(size_t));
      m_query_proofs.resize(nof_queries);
      for (size_t i = 0; i < nof_queries; ++i) {
        if (length < sizeof(size_t)) {
          ICICLE_LOG_ERROR << "Deserialization failed: length < sizeof(size_t)";
          return eIcicleError::COPY_FAILED;
        }
        size_t nof_fri_rounds;
        std::memcpy(&nof_fri_rounds, in, sizeof(size_t));
        advance(sizeof(size_t));
        m_query_proofs[i].resize(nof_fri_rounds);
        for (size_t j = 0; j < nof_fri_rounds; ++j) {
          eIcicleError err = m_query_proofs[i][j].deserialize(in, length);
          if (err != eIcicleError::SUCCESS) {
            ICICLE_LOG_ERROR << "Deserialization failed: m_query_proofs[" << i << "][" << j << "].deserialize(in, length)";
            return err;
          }
        }
      }
      size_t F_type;
      if (length < sizeof(size_t)) {
        ICICLE_LOG_ERROR << "Deserialization failed: length < sizeof(size_t)";
        return eIcicleError::COPY_FAILED;
      }
      std::memcpy(&F_type, in, sizeof(size_t));
      advance(sizeof(size_t));
      if (F_type != typeid(F).hash_code()) {
        ICICLE_LOG_ERROR << "Deserialization failed: F_type != typeid(F).hash_code()";
        return eIcicleError::INVALID_ARGUMENT;
      }

      if (length < sizeof(size_t)) {
        ICICLE_LOG_ERROR << "Deserialization failed: length < sizeof(size_t)";
        return eIcicleError::COPY_FAILED;
      }
      size_t final_poly_size;
      std::memcpy(&final_poly_size, in, sizeof(size_t));
      advance(sizeof(size_t));
      if (length < final_poly_size * sizeof(F)) {
        ICICLE_LOG_ERROR << "Deserialization failed: length < final_poly_size * sizeof(F)";
        return eIcicleError::COPY_FAILED;
      }
      m_final_poly.resize(final_poly_size);
      std::memcpy(m_final_poly.data(), in, final_poly_size * sizeof(F));
      advance(final_poly_size * sizeof(F));
      if (length < sizeof(uint64_t)) {
        ICICLE_LOG_ERROR << "Deserialization failed: length < sizeof(uint64_t)";
        return eIcicleError::COPY_FAILED;
      }
      std::memcpy(&m_pow_nonce, in, sizeof(uint64_t));
      advance(sizeof(uint64_t));
      return eIcicleError::SUCCESS;
    }
    
  private:
    std::vector<std::vector<MerkleProof>>
      m_query_proofs; // Matrix of Merkle proofs [query][round] - contains path, root, leaf. for each query, we have 2
                      // proofs (for the leaf in [2*query] and its symmetric in [2*query+1])
    std::vector<F> m_final_poly; // Final polynomial (constant in canonical FRI)
    uint64_t m_pow_nonce;        // Proof-of-work nonce
  };

} // namespace icicle