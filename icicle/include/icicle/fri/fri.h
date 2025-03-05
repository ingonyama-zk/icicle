#pragma once

#include <cstdint>
#include <memory>
#include <sys/types.h>
#include <vector>
#include "icicle/errors.h"
#include "icicle/backend/fri_backend.h"
#include "icicle/fri/fri_config.h"
#include "icicle/fri/fri_proof.h"
#include "icicle/fri/fri_transcript_config.h"
#include "icicle/hash/hash.h"
#include "icicle/merkle/merkle_tree.h"
#include "icicle/fri/fri_transcript.h"
#include "icicle/utils/log.h"

namespace icicle {

  /**
   * @brief Forward declaration for the FRI class template.
   */
  template <typename S, typename F>
  class Fri;

  /**
   * @brief Constructor for the case where only binary Merkle trees are used
   *        with a constant hash function.
   *
   * @param input_size The size of the input polynomial - number of evaluations.
   * @param folding_factor The factor by which the codeword is folded each round.
   * @param stopping_degree The minimal polynomial degree at which to stop folding.
   * @param merkle_tree_leaves_hash The hash function used for leaves of the Merkle tree.
   * @param merkle_tree_compress_hash The hash function used for compressing Merkle tree nodes.
   * @param output_store_min_layer (Optional) The layer at which to store partial results. Default = 0.
   * @return A `Fri<F>` object built around the chosen backend.
   */
  template <typename S, typename F>
  Fri<S, F> create_fri(
    const size_t input_size,
    const size_t folding_factor,
    const size_t stopping_degree,
    Hash merkle_tree_leaves_hash,
    Hash merkle_tree_compress_hash,
    const uint64_t output_store_min_layer = 0);

  /**
   * @brief Class for performing FRI operations.
   *
   * This class provides a high-level interface for constructing and managing a FRI proof.
   *
   * @tparam F The field type used in the FRI protocol.
   */
  template <typename S, typename F>
  class Fri
  {
  public:
    /**
     * @brief Constructor for the Fri class.
     * @param backend A shared pointer to the backend (FriBackend<S, F>) responsible for FRI operations.
     */
    explicit Fri(std::shared_ptr<FriBackend<S, F>> backend) : m_backend(std::move(backend)) {}

    /**
     * @brief Generate a FRI proof from the given polynomial evaluations (or input data).
     * @param fri_config Configuration for FRI operations (e.g., proof-of-work, queries).
     * @param fri_transcript_config Configuration for encoding/hashing (Fiat-Shamir).
     * @param input_data Evaluations or other relevant data for constructing the proof.
     * @param fri_proof Reference to a FriProof object (output).
     * @return An eIcicleError indicating success or failure.
     */
    eIcicleError get_proof(
      const FriConfig& fri_config,
      const FriTranscriptConfig<F>& fri_transcript_config,
      const F* input_data,
      FriProof<F>& fri_proof /* OUT */) const
    {
      return m_backend->get_proof(fri_config, fri_transcript_config, input_data, fri_proof);
    }

    /**
     * @brief Verify a FRI proof.
     * @param fri_config Configuration for FRI operations.
     * @param fri_transcript_config Configuration for encoding/hashing (Fiat-Shamir).
     * @param fri_proof The proof object to verify.
     * @param valid (OUT) Set to true if verification succeeds, false otherwise.
     * @return An eIcicleError indicating success or failure.
     */
    eIcicleError verify(
      const FriConfig& fri_config,
      const FriTranscriptConfig<F>& fri_transcript_config,
      FriProof<F>& fri_proof,
      bool& valid /* OUT */) const
    {
      if (__builtin_expect(fri_config.nof_queries <= 0, 0)) { ICICLE_LOG_ERROR << "Number of queries must be > 0"; }

      const size_t nof_fri_rounds = fri_proof.get_nof_fri_rounds();
      const size_t final_poly_size = fri_proof.get_final_poly_size();
      const uint32_t log_input_size = nof_fri_rounds + static_cast<uint32_t>(std::log2(final_poly_size));

      FriTranscript<F> transcript(fri_transcript_config, log_input_size);
      std::vector<F> alpha_values(nof_fri_rounds);
      eIcicleError err = update_transcript_and_generate_alphas_from_proof(fri_proof, transcript, nof_fri_rounds, alpha_values);
      if (err != eIcicleError::SUCCESS){ return err; }

      // Validate proof-of-work
      if (fri_config.pow_bits != 0) {
        bool pow_valid = false;
        check_pow_nonce_and_set_to_transcript(fri_proof, transcript, fri_config, pow_valid);
        if (!pow_valid) return eIcicleError::SUCCESS; // return with valid = false
      }

      // verify queries
      bool queries_valid = false;
      std::vector<size_t> queries_indicies =
      transcript.rand_queries_indicies(fri_config.nof_queries, final_poly_size, 1 << log_input_size, err);
      if (err != eIcicleError::SUCCESS) { return err; }
      err = verify_queries(fri_proof, fri_config.nof_queries, queries_indicies, alpha_values, queries_valid);
      if (!queries_valid) return eIcicleError::SUCCESS; // return with valid = false

      valid = true;
      return err;
    }

  private:
    std::shared_ptr<FriBackend<S, F>> m_backend;

    /**
     * @brief Updates the transcript with Merkle roots and generates alpha values for each round.
     * @param fri_proof The proof object containing Merkle roots.
     * @param transcript The transcript storing challenges.
     * @param nof_fri_rounds Number of FRI rounds.
     * @param alpha_values (OUT) Vector to store computed alpha values.
     */
    eIcicleError update_transcript_and_generate_alphas_from_proof(
      FriProof<F>& fri_proof,
      FriTranscript<F>& transcript,
      const size_t nof_fri_rounds,
      std::vector<F>& alpha_values) const
    {
      for (size_t round_idx = 0; round_idx < nof_fri_rounds; ++round_idx) {
        auto [root_ptr, root_size] = fri_proof.get_merkle_tree_root(round_idx);
        if (root_ptr == nullptr || root_size <= 0) {
          ICICLE_LOG_ERROR << "Failed to retrieve Merkle root for round " << round_idx;
        }
        std::vector<std::byte> merkle_commit(root_size);
        std::memcpy(merkle_commit.data(), root_ptr, root_size);
        eIcicleError err;
        alpha_values[round_idx] = transcript.get_alpha(merkle_commit, round_idx == 0, err);
        if (err != eIcicleError::SUCCESS){ return err; }
      }
      return eIcicleError::SUCCESS;
    }

    /**
     * @brief Validates the proof-of-work nonce from the fri_proof and sets it in the transcript.
     * @param fri_proof The proof containing the PoW nonce.
     * @param transcript The transcript where the nonce is recorded.
     * @param fri_config Configuration specifying required PoW bits.
     * @param pow_valid (OUT) Set to true if PoW verification succeeds.
     */
    void check_pow_nonce_and_set_to_transcript(
      FriProof<F>& fri_proof, FriTranscript<F>& transcript, const FriConfig& fri_config, bool& pow_valid) const
    {
      pow_valid = (transcript.hash_and_get_nof_leading_zero_bits(fri_proof.get_pow_nonce()) == fri_config.pow_bits);
      if (pow_valid) { transcript.set_pow_nonce(fri_proof.get_pow_nonce()); }
    }

    /**
     * @brief Checks if the leaf index from the proof matches the expected index computed based on the transcript random
     * generation.
     * @param leaf_index Index extracted from the proof.
     * @param leaf_index_sym Symmetric index extracted from the proof.
     * @param query The query based on the transcript random generation.
     * @param round_idx Current FRI round index.
     * @param log_input_size Log of the initial input size.
     * @return True if indices are consistent, false otherwise.
     */
    bool leaf_index_consistency_check(
      const uint64_t leaf_index,
      const uint64_t leaf_index_sym,
      const size_t query,
      const size_t round_idx,
      const uint32_t log_input_size) const
    {
      size_t round_size = (1ULL << (log_input_size - round_idx));
      size_t elem_idx = query % round_size;
      size_t elem_idx_sym = (query + (round_size >> 1)) % round_size;
      if (__builtin_expect(elem_idx != leaf_index, 0)) {
        ICICLE_LOG_ERROR << "Leaf index from proof doesn't match query expected index";
        return false;
      }
      if (__builtin_expect(elem_idx_sym != leaf_index_sym, 0)) {
        ICICLE_LOG_ERROR << "Leaf index symmetry from proof doesn't match query expected index";
        return false;
      }
      return true;
    }

    /**
     * @brief Validates collinearity in the folding process for a specific round.
     *        This ensures that the folded value computed from the queried elements
     *        matches the expected value in the proof.
     * @param fri_proof The proof object containing leaf data.
     * @param leaf_data Pointer to the leaf data.
     * @param leaf_data_sym Pointer to the symmetric leaf data.
     * @param query_idx Index of the query being verified.
     * @param query The query based on the transcript random generation.
     * @param round_idx Current FRI round index.
     * @param alpha_values Vector of alpha values for each round.
     * @param log_input_size Log of the initial input size.
     * @param primitive_root_inv Inverse primitive root used in calculations.
     * @return True if the collinearity check passes, false otherwise.
     */
    bool collinearity_check(
      FriProof<F>& fri_proof,
      const std::byte* leaf_data,
      const std::byte* leaf_data_sym,
      const size_t query_idx,
      const size_t query,
      const size_t round_idx,
      std::vector<F>& alpha_values,
      const uint32_t log_input_size,
      const S primitive_root_inv) const
    {
      const F& leaf_data_f = *reinterpret_cast<const F*>(leaf_data);
      const F& leaf_data_sym_f = *reinterpret_cast<const F*>(leaf_data_sym);
      size_t round_size = (1ULL << (log_input_size - round_idx));
      size_t elem_idx = query % round_size;
      F l_even = (leaf_data_f + leaf_data_sym_f) * S::inv_log_size(1);
      F l_odd = ((leaf_data_f - leaf_data_sym_f) * S::inv_log_size(1)) *
                S::pow(primitive_root_inv, elem_idx * (1 << round_idx));
      F alpha = alpha_values[round_idx];
      F folded = l_even + (alpha * l_odd);

      const size_t nof_fri_rounds = fri_proof.get_nof_fri_rounds();
      const size_t final_poly_size = fri_proof.get_final_poly_size();
      if (round_idx == nof_fri_rounds - 1) {
        const F* final_poly = fri_proof.get_final_poly();
        if (final_poly[query % final_poly_size] != folded) {
          ICICLE_LOG_ERROR << " (last round) Collinearity check failed for query=" << query
                           << ", query_idx=" << query_idx << ", round=" << round_idx;
          return false;
        }
      } else {
        MerkleProof& proof_ref_folded = fri_proof.get_query_proof(2 * query_idx, round_idx + 1);
        const auto [leaf_data_folded, leaf_size_folded, leaf_index_folded] = proof_ref_folded.get_leaf();
        const F& leaf_data_folded_f = *reinterpret_cast<const F*>(leaf_data_folded);
        if (leaf_data_folded_f != folded) {
          ICICLE_LOG_ERROR << "Collinearity check failed. query=" << query << ", query_idx=" << query_idx
                           << ", round=" << round_idx << ".\nfolded_res = \t\t" << folded << "\nfolded_from_proof = \t"
                           << leaf_data_folded_f;
          return false;
        }
      }
      return true;
    }

    /**
     * @brief Verifies Merkle proofs for a given query.
     * @param current_round_tree The Merkle tree corresponding to the round.
     * @param proof_ref Merkle proof for the query.
     * @param proof_ref_sym Merkle proof for the symmetric query.
     * @return True if both proofs are valid, false otherwise.
     */
    bool verify_merkle_proofs_for_query(
      const MerkleTree& current_round_tree, MerkleProof& proof_ref, MerkleProof& proof_ref_sym) const
    {
      bool merkle_proof_valid = false;
      eIcicleError err = current_round_tree.verify(proof_ref, merkle_proof_valid);
      if (err != eIcicleError::SUCCESS) {
        ICICLE_LOG_ERROR << "Merkle path verification returned err";
        return false;
      }
      if (!merkle_proof_valid) {
        ICICLE_LOG_ERROR << "Merkle path verification failed";
        return false;
      }

      merkle_proof_valid = false;
      eIcicleError err_sym = current_round_tree.verify(proof_ref_sym, merkle_proof_valid);
      if (err_sym != eIcicleError::SUCCESS) {
        ICICLE_LOG_ERROR << "Merkle path verification returned err";
        return false;
      }
      if (!merkle_proof_valid) {
        ICICLE_LOG_ERROR << "Merkle path sym verification failed";
        return false;
      }
      return true;
    }

    /**
     * @brief Verifies all queries in the FRI proof. This includes:
     *        - Checking Merkle proofs for consistency.
     *        - Ensuring leaf indices in the proof match those derived from the transcript.
     *        - Validating collinearity in the folding process.
     * @param fri_proof The proof object to verify.
     * @param nof_queries The number of queries.
     * @param queries_indices List of query indices to check.
     * @param alpha_values Vector of alpha values for each round.
     * @param queries_valid (OUT) Set to true if all queries pass verification.
     * @return An eIcicleError indicating success or failure.
     */

    eIcicleError verify_queries(
      FriProof<F>& fri_proof,
      const size_t nof_queries,
      std::vector<size_t>& queries_indicies,
      std::vector<F>& alpha_values,
      bool& queries_valid) const
    {
      const uint32_t log_input_size =
        fri_proof.get_nof_fri_rounds() + static_cast<uint32_t>(std::log2(fri_proof.get_final_poly_size()));
      S primitive_root_inv = S::omega_inv(log_input_size);
      for (size_t query_idx = 0; query_idx < nof_queries; query_idx++) {
        size_t query = queries_indicies[query_idx];
        for (size_t round_idx = 0; round_idx < fri_proof.get_nof_fri_rounds(); ++round_idx) {
          MerkleTree current_round_tree = m_backend->m_merkle_trees[round_idx];
          MerkleProof& proof_ref = fri_proof.get_query_proof(2 * query_idx, round_idx);
          MerkleProof& proof_ref_sym = fri_proof.get_query_proof(2 * query_idx + 1, round_idx);
          const auto [leaf_data, leaf_size, leaf_index] = proof_ref.get_leaf();
          const auto [leaf_data_sym, leaf_size_sym, leaf_index_sym] = proof_ref_sym.get_leaf();

          if (!verify_merkle_proofs_for_query(current_round_tree, proof_ref, proof_ref_sym)) {
            return eIcicleError::SUCCESS; // return with queries_valid = false
          }

          if (!leaf_index_consistency_check(leaf_index, leaf_index_sym, query, round_idx, log_input_size)) {
            return eIcicleError::SUCCESS; // return with queries_valid = false
          }

          if (!collinearity_check(
                fri_proof, leaf_data, leaf_data_sym, query_idx, query, round_idx, alpha_values, log_input_size,
                primitive_root_inv)) {
            return eIcicleError::SUCCESS; // return with queries_valid = false
          }
        }
      }
      queries_valid = true;
      return eIcicleError::SUCCESS;
    }
  };

} // namespace icicle
