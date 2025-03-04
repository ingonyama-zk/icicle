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
   * @brief Constructor for the case where Merkle trees are already given.
   *
   * @param folding_factor The factor by which the codeword is folded each round.
   * @param stopping_degree The minimal polynomial degree at which to stop folding.
   * @param merkle_trees A reference vector of `MerkleTree` objects.
   * @return A `Fri<F>` object built around the chosen backend.
   */
  template <typename S, typename F>
  Fri<S, F> create_fri(const size_t folding_factor, const size_t stopping_degree, std::vector<MerkleTree> merkle_trees);

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
      const FriTranscriptConfig<F>&& fri_transcript_config,
      const F* input_data,
      FriProof<F>& fri_proof /* OUT */) const
    {
      return m_backend->get_proof(fri_config, std::move(fri_transcript_config), input_data, fri_proof);
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
      const FriTranscriptConfig<F>&& fri_transcript_config,
      FriProof<F>& fri_proof,
      bool& valid /* OUT */) const
    {
      valid = false;
      if(__builtin_expect(fri_config.nof_queries <= 0, 0)){
        ICICLE_LOG_ERROR << "Number of queries must be > 0";
      }

      const size_t nof_fri_rounds = fri_proof.get_nof_fri_rounds();
      const size_t final_poly_size = fri_proof.get_final_poly_size();
      const uint32_t log_input_size = nof_fri_rounds + static_cast<uint32_t>(std::log2(final_poly_size));
      const size_t input_size = 1 << (log_input_size);
      std::vector<F> alpha_values(nof_fri_rounds);

      // set up the transcript
      FriTranscript<F> transcript(
        std::move(const_cast<FriTranscriptConfig<F>&>(fri_transcript_config)), log_input_size);
      for (size_t round_idx = 0; round_idx < nof_fri_rounds; ++round_idx) {
        auto [root_ptr, root_size] = fri_proof.get_merkle_tree_root(round_idx);
        if (root_ptr == nullptr || root_size <= 0){
          ICICLE_LOG_ERROR << "Failed to retrieve Merkle root for round " << round_idx;
        }
        std::vector<std::byte> merkle_commit(root_size);
        std::memcpy(merkle_commit.data(), root_ptr, root_size);
        alpha_values[round_idx] = transcript.get_alpha(merkle_commit);
      }

      // proof-of-work
      if (fri_config.pow_bits != 0) {
        bool valid = (transcript.hash_and_get_nof_leading_zero_bits(fri_proof.get_pow_nonce()) == fri_config.pow_bits);
        if (!valid) return eIcicleError::SUCCESS; // return with valid = false
        transcript.set_pow_nonce(fri_proof.get_pow_nonce());
      }

      // get query indices
      size_t seed = transcript.get_seed_for_query_phase();
      seed_rand_generator(seed);
      std::vector<size_t> query_indices = rand_size_t_vector(fri_config.nof_queries, final_poly_size, input_size);

      uint64_t domain_max_size = 0;
      uint64_t max_log_size = 0;
      S primitive_root_inv = S::omega_inv(log_input_size);

      for (size_t query_idx = 0; query_idx < fri_config.nof_queries; query_idx++) {
        size_t query = query_indices[query_idx];
        size_t current_log_size = log_input_size;
        for (size_t round_idx = 0; round_idx < nof_fri_rounds; ++round_idx) {
          size_t round_size = (1ULL << (log_input_size - round_idx));
          size_t elem_idx = query % round_size;
          size_t elem_idx_sym = (query + (round_size >> 1)) % round_size;

          MerkleTree current_round_tree = m_backend->m_merkle_trees[round_idx];
          MerkleProof& proof_ref = fri_proof.get_query_proof(2 * query_idx, round_idx);
          bool valid = false;
          eIcicleError err = current_round_tree.verify(proof_ref, valid);
          if (err != eIcicleError::SUCCESS) {
            ICICLE_LOG_ERROR << "[VERIFIER] Merkle path verification returned err for query=" << query
                             << ", query_idx=" << query_idx << ", round=" << round_idx;
            return err;
          }
          if (!valid) {
            ICICLE_LOG_ERROR << "[VERIFIER] Merkle path verification failed for leaf query=" << query
                             << ", query_idx=" << query_idx << ", round=" << round_idx;
            return eIcicleError::SUCCESS; // return with valid = false
          }

          MerkleProof& proof_ref_sym = fri_proof.get_query_proof(2 * query_idx + 1, round_idx);
          valid = false;
          eIcicleError err_sym = current_round_tree.verify(proof_ref_sym, valid);
          if (err_sym != eIcicleError::SUCCESS) {
            ICICLE_LOG_ERROR << "Merkle path verification returned err for query=" << query
                             << ", query_idx=" << query_idx << ", round=" << round_idx;
            return err_sym;
          }
          if (!valid) {
            ICICLE_LOG_ERROR << "Merkle path verification failed for leaf query=" << query
                             << ", query_idx=" << query_idx << ", round=" << round_idx;
            return eIcicleError::SUCCESS; // return with valid = false
          }

          // collinearity check
          const auto [leaf_data, leaf_size, leaf_index] = proof_ref.get_leaf();
          const auto [leaf_data_sym, leaf_size_sym, leaf_index_sym] = proof_ref_sym.get_leaf();
          if(__builtin_expect(elem_idx != leaf_index, 0)){
            ICICLE_LOG_ERROR << "Leaf index from proof doesn't match query expected index";
          }
          if(__builtin_expect(elem_idx_sym != leaf_index_sym, 0)){
            ICICLE_LOG_ERROR << "Leaf index symmetry from proof doesn't match query expected index";
          }
          const F& leaf_data_f = *reinterpret_cast<const F*>(leaf_data);
          const F& leaf_data_sym_f = *reinterpret_cast<const F*>(leaf_data_sym);
          F l_even = (leaf_data_f + leaf_data_sym_f) * S::inv_log_size(1);
          F l_odd = ((leaf_data_f - leaf_data_sym_f) * S::inv_log_size(1)) *
                    S::pow(primitive_root_inv, leaf_index * (input_size >> current_log_size));
          F alpha = alpha_values[round_idx];
          F folded = l_even + (alpha * l_odd);

          if (round_idx == nof_fri_rounds - 1) {
            const F* final_poly = fri_proof.get_final_poly();
            if (final_poly[query % final_poly_size] != folded) {
              ICICLE_LOG_ERROR << "[VERIFIER] (last round) Collinearity check failed for query=" << query
                               << ", query_idx=" << query_idx << ", round=" << round_idx;
              return eIcicleError::SUCCESS; // return with valid = false;
            }
          } else {
            MerkleProof& proof_ref_folded = fri_proof.get_query_proof(2 * query_idx, round_idx + 1);
            const auto [leaf_data_folded, leaf_size_folded, leaf_index_folded] = proof_ref_folded.get_leaf();
            const F& leaf_data_folded_f = *reinterpret_cast<const F*>(leaf_data_folded);
            if (leaf_data_folded_f != folded) {
              ICICLE_LOG_ERROR << "[VERIFIER] Collinearity check failed. query=" << query << ", query_idx=" << query_idx
                               << ", round=" << round_idx << ".\nfolded_res = \t\t" << folded
                               << "\nfolded_from_proof = \t" << leaf_data_folded_f;
              return eIcicleError::SUCCESS; // return with valid = false
            }
          }
          current_log_size--;
        }
      }
      valid = true;
      return eIcicleError::SUCCESS;
    }

  private:
    std::shared_ptr<FriBackend<S, F>> m_backend;
  };

} // namespace icicle
