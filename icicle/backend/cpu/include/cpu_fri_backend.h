#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <random>
#include "icicle/errors.h"
#include "icicle/fri/fri_transcript.h"
#include "icicle/backend/fri_backend.h"
#include "cpu_fri_rounds.h"
#include "cpu_ntt_domain.h"
#include "icicle/utils/log.h"

namespace icicle {
  template <typename S, typename F>
  class CpuFriBackend : public FriBackend<S, F>
  {
  public:
    /**
     * @brief Constructor that accepts an existing array of Merkle trees.
     *
     * @param folding_factor   The factor by which the codeword is folded each round.
     * @param stopping_degree  Stopping degree threshold for the final polynomial.
     * @param merkle_trees     A vector of MerkleTrees, tree per FRI round.
     */
    CpuFriBackend(const size_t folding_factor, const size_t stopping_degree, std::vector<MerkleTree> merkle_trees)
        : FriBackend<S, F>(folding_factor, stopping_degree, merkle_trees), m_nof_fri_rounds(merkle_trees.size()),
          m_log_input_size(merkle_trees.size() + std::log2(static_cast<double>(stopping_degree + 1))),
          m_input_size(pow(2, m_log_input_size)), m_fri_rounds(merkle_trees, m_log_input_size)
    {
    }

    eIcicleError get_fri_proof(
      const FriConfig& fri_config,
      const FriTranscriptConfig<F>&& fri_transcript_config,
      const F* input_data,
      FriProof<F>& fri_proof /*out*/) override
    {
      if(__builtin_expect(fri_config.nof_queries <= 0, 0)){
        ICICLE_LOG_ERROR << "Number of queries must be > 0";
      }

      FriTranscript<F> transcript(std::move(fri_transcript_config), m_log_input_size);

      // Initialize the proof
      fri_proof.init(fri_config.nof_queries, m_nof_fri_rounds, this->m_stopping_degree + 1);

      // commit fold phase
      ICICLE_CHECK(commit_fold_phase(input_data, transcript, fri_config, fri_proof));

      // proof of work
      if (fri_config.pow_bits != 0) { ICICLE_CHECK(proof_of_work(transcript, fri_config.pow_bits, fri_proof)); }

      // query phase
      ICICLE_CHECK(query_phase(transcript, fri_config, fri_proof));

      return eIcicleError::SUCCESS;
    }

  private:
    const size_t m_nof_fri_rounds; // Number of FRI rounds
    const size_t m_log_input_size; // Log size of the input polynomial
    const size_t m_input_size;     // Size of the input polynomial
    FriRounds<F> m_fri_rounds;     // Holds intermediate rounds

    /**
     * @brief Perform the commit-fold phase of the FRI protocol.
     *
     * @param input_data  The initial polynomial evaluations.
     * @param fri_proof   The proof object to update.
     * @param transcript  The transcript to generate challenges.
     * @return eIcicleError Error code indicating success or failure.
     */
    eIcicleError commit_fold_phase(
      const F* input_data, FriTranscript<F>& transcript, const FriConfig& fri_config, FriProof<F>& fri_proof)
    {
      if (this->m_folding_factor != 2){
        ICICLE_LOG_ERROR << "Currently only folding factor of 2 is supported"; // TODO SHANIE - remove when supporting other folding
        // factors
      }        
      const S* twiddles = ntt_cpu::CpuNttDomain<S>::s_ntt_domain.get_twiddles();
      uint64_t domain_max_size = ntt_cpu::CpuNttDomain<S>::s_ntt_domain.get_max_size();

      // Get persistent storage for round from FriRounds. m_fri_rounds already allocated a vector for each round with
      // capacity 2^(m_log_input_size - round_idx).
      F* round_evals = m_fri_rounds.get_round_evals(0);
      std::copy(input_data, input_data + m_input_size, round_evals);

      size_t current_size = m_input_size;
      size_t current_log_size = m_log_input_size;

      for (size_t round_idx = 0; round_idx < m_nof_fri_rounds; ++round_idx) {
        // Merkle tree for the current round_idx
        MerkleTree* current_round_tree = m_fri_rounds.get_merkle_tree(round_idx);
        current_round_tree->build(round_evals, current_size, MerkleTreeConfig());
        auto [root_ptr, root_size] = current_round_tree->get_merkle_root();
        if (root_ptr == nullptr || root_size <= 0){
          ICICLE_LOG_ERROR << "Failed to retrieve Merkle root for round " << round_idx;
        }
        // Add root to transcript and get alpha
        std::vector<std::byte> merkle_commit(root_size);
        std::memcpy(merkle_commit.data(), root_ptr, root_size);

        F alpha = transcript.get_alpha(merkle_commit);

        // Fold the evaluations
        size_t half = current_size >> 1;
        std::vector<F> peven(half);
        std::vector<F> podd(half);

        for (size_t i = 0; i < half; ++i) {
          peven[i] = (round_evals[i] + round_evals[i + half]) * S::inv_log_size(1);
          uint64_t tw_idx = domain_max_size - ((domain_max_size >> current_log_size) * i);
          podd[i] = ((round_evals[i] - round_evals[i + half]) * S::inv_log_size(1)) * twiddles[tw_idx];
        }

        if (round_idx == m_nof_fri_rounds - 1) {
          round_evals = fri_proof.get_final_poly();
        } else {
          round_evals = m_fri_rounds.get_round_evals(round_idx + 1);
        }

        for (size_t i = 0; i < half; ++i) {
          round_evals[i] = peven[i] + (alpha * podd[i]);
        }

        current_size >>= 1;
        current_log_size--;
      }
      return eIcicleError::SUCCESS;
    }

    eIcicleError proof_of_work(FriTranscript<F>& transcript, const size_t pow_bits, FriProof<F>& fri_proof)
    {
      for (uint64_t nonce = 0; nonce < UINT64_MAX; nonce++) {
        if (transcript.hash_and_get_nof_leading_zero_bits(nonce) == pow_bits) {
          transcript.set_pow_nonce(nonce);
          fri_proof.set_pow_nonce(nonce);
          return eIcicleError::SUCCESS;
        }
      }
      ICICLE_LOG_ERROR << "Failed to find a proof-of-work nonce";
      return eIcicleError::UNKNOWN_ERROR;
    }

    /**
     * @brief Perform the query phase of the FRI protocol.
     *
     * @param transcript      The transcript object.
     * @param fri_config      The FRI configuration object.
     * @param fri_proof       (OUT) The proof object where we store the resulting Merkle proofs.
     * @return eIcicleError
     */
    eIcicleError query_phase(FriTranscript<F>& transcript, const FriConfig& fri_config, FriProof<F>& fri_proof)
    {
      size_t seed = transcript.get_seed_for_query_phase();
      seed_rand_generator(seed);
      std::vector<size_t> query_indices =
        rand_size_t_vector(fri_config.nof_queries, (this->m_stopping_degree + 1), m_input_size);

      for (size_t query_idx = 0; query_idx < fri_config.nof_queries; query_idx++) {
        size_t query = query_indices[query_idx];
        for (size_t round_idx = 0; round_idx < m_nof_fri_rounds; round_idx++) {
          size_t round_size = (1ULL << (m_log_input_size - round_idx));
          size_t leaf_idx = query % round_size;
          size_t leaf_idx_sym = (query + (round_size >> 1)) % round_size;
          F* round_evals = m_fri_rounds.get_round_evals(round_idx);

          MerkleProof& proof_ref = fri_proof.get_query_proof(2 * query_idx, round_idx);
          eIcicleError err = m_fri_rounds.get_merkle_tree(round_idx)->get_merkle_proof(
            round_evals, round_size, leaf_idx, false /* is_pruned */, MerkleTreeConfig(), proof_ref);
          if (err != eIcicleError::SUCCESS) return err;
          MerkleProof& proof_ref_sym = fri_proof.get_query_proof(2 * query_idx + 1, round_idx);
          eIcicleError err_sym = m_fri_rounds.get_merkle_tree(round_idx)->get_merkle_proof(
            round_evals, round_size, leaf_idx_sym, false /* is_pruned */, MerkleTreeConfig(), proof_ref_sym);
          if (err_sym != eIcicleError::SUCCESS) return err_sym;
        }
      }
      return eIcicleError::SUCCESS;
    }
  };

} // namespace icicle
