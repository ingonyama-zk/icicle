#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <random>
#include "icicle/errors.h"
#include "cpu_fri_transcript.h"
#include "icicle/backend/fri_backend.h"
#include "cpu_fri_transcript.h"
#include "cpu_fri_rounds.h"
#include "cpu_ntt_domain.h"
#include "icicle/utils/log.h"

namespace icicle {
  template <typename F>
  class CpuFriBackend : public FriBackend<F>
  {
  public:
    /**
     * @brief Constructor that accepts an existing array of Merkle trees.
     *
     * @param folding_factor   The factor by which the codeword is folded each round.
     * @param stopping_degree  Stopping degree threshold for the final polynomial.
     * @param merkle_trees     A moved vector of MerkleTree pointers.
     */
    CpuFriBackend(const size_t folding_factor, const size_t stopping_degree, std::vector<MerkleTree>&& merkle_trees)
        : FriBackend<F>(folding_factor, stopping_degree),
          m_log_input_size(merkle_trees.size()),
          m_input_size(pow(2, m_log_input_size)),
          m_fri_rounds(std::move(merkle_trees))
    {
    }

    eIcicleError get_fri_proof(
      const FriConfig& fri_config,
      const FriTranscriptConfig<F>& fri_transcript_config,
      const F* input_data,
      FriProof<F>& fri_proof /*out*/) override
    {
      if (fri_config.use_extension_field) {
        ICICLE_LOG_ERROR << "FriConfig::use_extension_field = true is currently unsupported";
        return eIcicleError::API_NOT_IMPLEMENTED;
      }
      ICICLE_ASSERT(fri_config.nof_queries > 0) << "Number of queries must be > 0";

      CpuFriTranscript<F> transcript(std::move(const_cast<FriTranscriptConfig<F>&>(fri_transcript_config)), m_log_input_size);

      // Determine the number of folding rounds
      size_t df = this->m_stopping_degree;
      size_t log_df_plus_1 = (df > 0) ? static_cast<size_t>(std::log2(static_cast<double>(df + 1))) : 0;
      size_t nof_fri_rounds = (m_log_input_size > log_df_plus_1) ? (m_log_input_size - log_df_plus_1) : 0;

      // Initialize the proof
      fri_proof.init(fri_config.nof_queries, nof_fri_rounds);

      //commit fold phase
      ICICLE_CHECK(commit_fold_phase(input_data, transcript, fri_config, nof_fri_rounds, fri_proof));

      //proof of work
      if (fri_config.pow_bits != 0){
        ICICLE_CHECK(proof_of_work(transcript, fri_config.pow_bits, fri_proof));
      }

      //query phase
      ICICLE_CHECK(query_phase(transcript, fri_config, nof_fri_rounds, fri_proof));

      return eIcicleError::SUCCESS;
    }

  private:
    FriRounds<F> m_fri_rounds;                  // Holds intermediate rounds
    const size_t m_log_input_size;              // Log size of the input polynomial
    const size_t m_input_size;                  // Size of the input polynomial

    /**
     * @brief Perform the commit-fold phase of the FRI protocol.
     *
     * @param input_data  The initial polynomial evaluations.
     * @param fri_proof   The proof object to update.
     * @param transcript  The transcript to generate challenges.
     * @return eIcicleError Error code indicating success or failure.
     */
    eIcicleError commit_fold_phase(const F* input_data, CpuFriTranscript<F>& transcript, const FriConfig& fri_config, size_t nof_fri_rounds, FriProof<F>& fri_proof){
      ICICLE_ASSERT(this->m_folding_factor==2) << "Folding factor must be 2";

      const F* twiddles = ntt_cpu::CpuNttDomain<F>::s_ntt_domain.get_twiddles();
      uint64_t domain_max_size = ntt_cpu::CpuNttDomain<F>::s_ntt_domain.get_max_size();

      // Get persistent storage for round from FriRounds. m_fri_rounds already allocated a vector for each round with capacity 2^(m_log_input_size - round_idx).
      F* round_evals = m_fri_rounds.get_round_evals(0);
      std::copy(input_data, input_data + m_input_size, round_evals);

      size_t current_size = m_input_size;
      size_t current_log_size = m_log_input_size;

      for (size_t round_idx = 0; round_idx < nof_fri_rounds; ++round_idx){
        // if (current_size == (df + 1)) { FIXME SHANIE - do I need this?
        //   fri_proof.finalpoly->assign(round_evals->begin(), round_evals->end());
        //   break;
        // }

        // Merkle tree for the current round_idx
        MerkleTree* current_round_tree = m_fri_rounds.get_merkle_tree(round_idx);
        current_round_tree->build(reinterpret_cast<const std::byte*>(round_evals), sizeof(F), MerkleTreeConfig());
        auto [root_ptr, root_size] = current_round_tree->get_merkle_root();
        ICICLE_ASSERT(root_ptr != nullptr && root_size > 0) << "Failed to retrieve Merkle root for round " << round_idx;

        // FIXME SHANIE - do I need to add the root to the proof here?

        // Add root to transcript and get alpha
        // std::vector<std::byte> merkle_commit(root_ptr, root_ptr + root_size); //FIXME SHANIE - what is the right way to convert?
        std::vector<std::byte> merkle_commit(root_size);
        std::memcpy(merkle_commit.data(), root_ptr, root_size);

        F alpha = transcript.get_alpha(merkle_commit);

        // Fold the evaluations
        size_t half = current_size>>1;
        std::vector<F> peven(half);
        std::vector<F> podd(half);

        for (size_t i = 0; i < half; ++i){
          peven[i] = (round_evals[i] + round_evals[i + half]) * F::inv_log_size(1);
          uint64_t tw_idx = domain_max_size - ((domain_max_size>>current_log_size) * i);
          podd[i] = ((round_evals[i] - round_evals[i + half]) * F::inv_log_size(1)) * twiddles[tw_idx];
        }

        if (round_idx == nof_fri_rounds - 1){
          round_evals = fri_proof.get_final_poly();
        } else {
          round_evals = m_fri_rounds.get_round_evals(round_idx + 1);
        }

        for (size_t i = 0; i < half; ++i){
          round_evals[i] = peven[i] + (alpha * podd[i]);
        }
        
        current_size>>=1;
        current_log_size--;
      }

      return eIcicleError::SUCCESS;
    }

    eIcicleError proof_of_work(CpuFriTranscript<F>& transcript, const size_t pow_bits, FriProof<F>& fri_proof){
      for (uint64_t nonce = 0; nonce < UINT64_MAX; nonce++)
      {
        if(transcript.hash_and_get_nof_leading_zero_bits(nonce) == pow_bits){
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
    eIcicleError query_phase(CpuFriTranscript<F>& transcript, const FriConfig& fri_config, size_t nof_fri_rounds, FriProof<F>& fri_proof)
    {
      ICICLE_ASSERT(fri_config.nof_queries > 0) << "Number of queries must be > 0";
      size_t seed = transcript.get_seed_for_query_phase();
      seed_rand_generator(seed);
      std::vector<size_t> query_indices = rand_size_t_vector(fri_config.nof_queries, (this->m_stopping_degree + 1), m_input_size);

      for (size_t q = 0; q < query_indices.size(); q++){
        size_t query = query_indices[q];
        for (size_t round_idx = 0; round_idx < nof_fri_rounds; round_idx++){
          size_t round_size = (1ULL << (m_log_input_size - round_idx));
          size_t query_idx = query % round_size;
          size_t query_idx_sym = (query + (round_size >> 1)) % round_size;
          F* round_evals = m_fri_rounds.get_round_evals(round_idx);
          const std::byte* leaves = reinterpret_cast<const std::byte*>(round_evals);
          uint64_t leaves_size = sizeof(F);

          MerkleProof& proof_ref = fri_proof.get_query_proof(query_idx, round_idx);              
          eIcicleError err = m_fri_rounds.get_merkle_tree(round_idx)->get_merkle_proof(leaves, sizeof(F), query_idx, false /* is_pruned */, MerkleTreeConfig(), proof_ref);
          if (err != eIcicleError::SUCCESS) return err;
          MerkleProof& proof_ref_sym = fri_proof.get_query_proof(query_idx_sym, round_idx);              
          eIcicleError err_sym = m_fri_rounds.get_merkle_tree(round_idx)->get_merkle_proof(leaves, sizeof(F), query_idx_sym, false /* is_pruned */, MerkleTreeConfig(), proof_ref_sym);
          if (err_sym != eIcicleError::SUCCESS) return err_sym;
        }
      }
      return eIcicleError::SUCCESS;
    }
  };

} // namespace icicle
