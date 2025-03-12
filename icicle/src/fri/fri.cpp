#include "icicle/fri/fri.h"
#include "icicle/dispatcher.h"
#include "icicle/errors.h"

namespace icicle {

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
     * @param input_data Evaluations of the input polynomial.
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
      const FriProof<F>& fri_proof,
      bool& valid /* OUT */) const
    {
      if (__builtin_expect(fri_config.nof_queries <= 0, 0)) { ICICLE_LOG_ERROR << "Number of queries must be > 0"; }

      const size_t nof_fri_rounds = fri_proof.get_nof_fri_rounds();
      const size_t final_poly_size = fri_proof.get_final_poly_size();
      const uint32_t log_input_size = nof_fri_rounds + static_cast<uint32_t>(std::log2(final_poly_size));

      FriTranscript<F> transcript(fri_transcript_config, log_input_size);
      std::vector<F> alpha_values(nof_fri_rounds);
      eIcicleError err =
        update_transcript_and_generate_alphas_from_proof(fri_proof, transcript, nof_fri_rounds, alpha_values);
      if (err != eIcicleError::SUCCESS) { return err; }

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
      const FriProof<F>& fri_proof,
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
        if (err != eIcicleError::SUCCESS) { return err; }
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
      const FriProof<F>& fri_proof, FriTranscript<F>& transcript, const FriConfig& fri_config, bool& pow_valid) const
    {
      uint64_t proof_pow_nonce = fri_proof.get_pow_nonce();
      pow_valid = transcript.verify_pow(proof_pow_nonce, fri_config.pow_bits);
      transcript.set_pow_nonce(proof_pow_nonce);
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
      const FriProof<F>& fri_proof,
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
        const MerkleProof& proof_ref_folded = fri_proof.get_query_proof_slot(2 * query_idx, round_idx + 1);
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
      const MerkleTree& current_round_tree, const MerkleProof& proof_ref, const MerkleProof& proof_ref_sym) const
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
      const FriProof<F>& fri_proof,
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
          const MerkleProof& proof_ref = fri_proof.get_query_proof_slot(2 * query_idx, round_idx);
          const MerkleProof& proof_ref_sym = fri_proof.get_query_proof_slot(2 * query_idx + 1, round_idx);
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

  using FriFactoryScalar = FriFactoryImpl<scalar_t, scalar_t>;
  ICICLE_DISPATCHER_INST(FriDispatcher, fri_factory, FriFactoryScalar);

  /**
   * @brief Create a FRI instance.
   * @return A `Fri<scalar_t, scalar_t>` object built around the chosen backend.
   */
  template <typename S, typename F>
  Fri<S, F> create_fri(
    const size_t log_input_size,
    const size_t folding_factor,
    const size_t stopping_degree,
    Hash merkle_tree_leaves_hash,
    Hash merkle_tree_compress_hash,
    const uint64_t output_store_min_layer)
  {
    const size_t df = stopping_degree;
    const size_t log_df_plus_1 = (df > 0) ? static_cast<size_t>(std::log2(static_cast<double>(df + 1))) : 0;
    const size_t fold_rounds = (log_input_size > log_df_plus_1) ? (log_input_size - log_df_plus_1) : 0;

    std::vector<MerkleTree> merkle_trees;
    merkle_trees.reserve(fold_rounds);
    size_t compress_hash_arity =
      merkle_tree_compress_hash.default_input_chunk_size() / merkle_tree_compress_hash.output_size();
    size_t first_merkle_tree_height = std::ceil(log_input_size / std::log2(compress_hash_arity)) + 1;
    std::vector<Hash> layer_hashes(first_merkle_tree_height, merkle_tree_compress_hash);
    layer_hashes[0] = merkle_tree_leaves_hash;
    uint64_t leaf_element_size = merkle_tree_leaves_hash.default_input_chunk_size();
    for (size_t i = 0; i < fold_rounds; i++) {
      merkle_trees.emplace_back(MerkleTree::create(layer_hashes, leaf_element_size, output_store_min_layer));
      layer_hashes.pop_back();
    }
    std::shared_ptr<FriBackend<S, F>> backend;
    ICICLE_CHECK(FriDispatcher::execute(
      folding_factor, stopping_degree, merkle_trees,
      backend)); // The MerkleTree class only holds a shared_ptr to MerkleTreeBackend, so copying is lightweight.

    Fri<S, F> fri{backend};
    return fri;
  }

#ifdef EXT_FIELD
  using FriExtFactoryScalar = FriFactoryImpl<scalar_t, extension_t>;
  ICICLE_DISPATCHER_INST(FriExtFieldDispatcher, extension_fri_factory, FriExtFactoryScalar);

  /**
   * @brief Create a FRI instance.
   * @return A `Fri<scalar_t, extension_t>` object built around the chosen backend.
   */
  template <typename S, typename F>
  Fri<S, F> create_fri_ext(
    const size_t log_input_size,
    const size_t folding_factor,
    const size_t stopping_degree,
    Hash merkle_tree_leaves_hash,
    Hash merkle_tree_compress_hash,
    const uint64_t output_store_min_layer)
  {
    const size_t df = stopping_degree;
    const size_t log_df_plus_1 = (df > 0) ? static_cast<size_t>(std::log2(static_cast<double>(df + 1))) : 0;
    const size_t fold_rounds = (log_input_size > log_df_plus_1) ? (log_input_size - log_df_plus_1) : 0;

    std::vector<MerkleTree> merkle_trees;
    merkle_trees.reserve(fold_rounds);
    size_t compress_hash_arity =
      merkle_tree_compress_hash.default_input_chunk_size() / merkle_tree_compress_hash.output_size();
    size_t first_merkle_tree_height = std::ceil(log_input_size / std::log2(compress_hash_arity)) + 1;
    std::vector<Hash> layer_hashes(first_merkle_tree_height, merkle_tree_compress_hash);
    layer_hashes[0] = merkle_tree_leaves_hash;
    uint64_t leaf_element_size = merkle_tree_leaves_hash.default_input_chunk_size();
    for (size_t i = 0; i < fold_rounds; i++) {
      merkle_trees.emplace_back(MerkleTree::create(layer_hashes, leaf_element_size, output_store_min_layer));
      layer_hashes.pop_back();
    }
    std::shared_ptr<FriBackend<S, F>> backend;
    ICICLE_CHECK(FriExtFieldDispatcher::execute(folding_factor, stopping_degree, merkle_trees, backend));

    Fri<S, F> fri{backend};
    return fri;
  }

#endif // EXT_FIELD

  eIcicleError check_if_valid(
    const size_t nof_queries, const size_t input_size, const size_t folding_factor, const size_t compress_hash_arity)
  {
    if (nof_queries <= 0 || nof_queries > (input_size / folding_factor)) {
      ICICLE_LOG_ERROR << "Number of queries must be > 0 and < input_size/folding_factor";
      return eIcicleError::INVALID_ARGUMENT;
    }
    if (folding_factor != 2) {
      ICICLE_LOG_ERROR << "Currently only folding factor of 2 is supported"; // TODO SHANIE (future) - remove when
                                                                             // supporting other folding factors
      return eIcicleError::INVALID_ARGUMENT;
    }
    if (input_size == 0 || (input_size & (input_size - 1)) != 0) {
      ICICLE_LOG_ERROR << "input_size must be a power of 2. input_size = " << input_size;
      return eIcicleError::INVALID_ARGUMENT;
    }
    if (folding_factor % compress_hash_arity != 0) {
      ICICLE_LOG_ERROR << "folding_factor must be divisible by compress_hash_arity. "
                       << "folding_factor = " << folding_factor << ", compress_hash_arity = " << compress_hash_arity;
      return eIcicleError::INVALID_ARGUMENT;
    }
    if (compress_hash_arity != 2) {
      ICICLE_LOG_ERROR << "Currently only compress hash arity of 2 is supported";
      return eIcicleError::INVALID_ARGUMENT;
    }
    return eIcicleError::SUCCESS;
  }

  template <>
  eIcicleError prove_fri_merkle_tree<scalar_t>(
    const FriConfig& fri_config,
    const FriTranscriptConfig<scalar_t>& fri_transcript_config,
    const scalar_t* input_data,
    const size_t input_size,
    Hash merkle_tree_leaves_hash,
    Hash merkle_tree_compress_hash,
    const uint64_t output_store_min_layer,
    FriProof<scalar_t>& fri_proof /* OUT */)
  {
    eIcicleError err = check_if_valid(
      fri_config.nof_queries, input_size, fri_config.folding_factor,
      merkle_tree_compress_hash.default_input_chunk_size() / merkle_tree_compress_hash.output_size());
    if (err != eIcicleError::SUCCESS) { return err; }
    const size_t log_input_size = std::log2(input_size);
    Fri prover_fri = create_fri<scalar_t, scalar_t>(
      log_input_size, fri_config.folding_factor, fri_config.stopping_degree, merkle_tree_leaves_hash,
      merkle_tree_compress_hash, output_store_min_layer);
    return prover_fri.get_proof(fri_config, fri_transcript_config, input_data, fri_proof);
  }

  extern "C" {
    eIcicleError CONCAT_EXPAND(FIELD, get_fri_proof_mt)(
      const FriConfig& fri_config,
      const FriTranscriptConfig<scalar_t>& fri_transcript_config,
      const scalar_t* input_data,
      const size_t input_size,
      Hash& merkle_tree_leaves_hash,
      Hash& merkle_tree_compress_hash,
      const uint64_t output_store_min_layer,
      FriProof<scalar_t>& fri_proof /* OUT */)
    {
      return get_fri_proof_mt<scalar_t, scalar_t>(
        fri_config,
        fri_transcript_config,
        input_data,
        input_size,
        merkle_tree_leaves_hash,
        merkle_tree_compress_hash,
        output_store_min_layer,
        fri_proof
      );
    }
  }

  template <>
  eIcicleError verify_fri_merkle_tree<scalar_t>(
    const FriConfig& fri_config,
    const FriTranscriptConfig<scalar_t>& fri_transcript_config,
    const FriProof<scalar_t>& fri_proof,
    Hash merkle_tree_leaves_hash,
    Hash merkle_tree_compress_hash,
    bool& valid /* OUT */)
  {
    const size_t nof_fri_rounds = fri_proof.get_nof_fri_rounds();
    const size_t final_poly_size = fri_proof.get_final_poly_size();
    const uint32_t log_input_size = nof_fri_rounds + static_cast<uint32_t>(std::log2(final_poly_size));

    size_t compress_hash_arity =
      merkle_tree_compress_hash.default_input_chunk_size() / merkle_tree_compress_hash.output_size();
    eIcicleError err =
      check_if_valid(fri_config.nof_queries, (1 << log_input_size), fri_config.folding_factor, compress_hash_arity);
    if (err != eIcicleError::SUCCESS) { return err; }
    Fri verifier_fri = create_fri<scalar_t, scalar_t>(
      log_input_size, fri_config.folding_factor, fri_config.stopping_degree, merkle_tree_leaves_hash,
      merkle_tree_compress_hash, 0);
    return verifier_fri.verify(fri_config, fri_transcript_config, fri_proof, valid);
  }

#ifdef EXT_FIELD
  template <>
  eIcicleError prove_fri_merkle_tree<extension_t>(
    const FriConfig& fri_config,
    const FriTranscriptConfig<extension_t>& fri_transcript_config,
    const extension_t* input_data,
    const size_t input_size,
    Hash merkle_tree_leaves_hash,
    Hash merkle_tree_compress_hash,
    const uint64_t output_store_min_layer,
    FriProof<extension_t>& fri_proof /* OUT */)
  {
    size_t compress_hash_arity =
      merkle_tree_compress_hash.default_input_chunk_size() / merkle_tree_compress_hash.output_size();
    eIcicleError err =
      check_if_valid(fri_config.nof_queries, input_size, fri_config.folding_factor, compress_hash_arity);
    if (err != eIcicleError::SUCCESS) { return err; }
    const size_t log_input_size = std::log2(input_size);
    Fri prover_fri = create_fri_ext<scalar_t, extension_t>(
      log_input_size, fri_config.folding_factor, fri_config.stopping_degree, merkle_tree_leaves_hash,
      merkle_tree_compress_hash, output_store_min_layer);
    return prover_fri.get_proof(fri_config, fri_transcript_config, input_data, fri_proof);
  }

  template <>
  eIcicleError verify_fri_merkle_tree<extension_t>(
    const FriConfig& fri_config,
    const FriTranscriptConfig<extension_t>& fri_transcript_config,
    const FriProof<extension_t>& fri_proof,
    Hash merkle_tree_leaves_hash,
    Hash merkle_tree_compress_hash,
    bool& valid /* OUT */)
  {
    const size_t nof_fri_rounds = fri_proof.get_nof_fri_rounds();
    const size_t final_poly_size = fri_proof.get_final_poly_size();
    const uint32_t log_input_size = nof_fri_rounds + static_cast<uint32_t>(std::log2(final_poly_size));

    size_t compress_hash_arity =
      merkle_tree_compress_hash.default_input_chunk_size() / merkle_tree_compress_hash.output_size();
    eIcicleError err =
      check_if_valid(fri_config.nof_queries, (1 << log_input_size), fri_config.folding_factor, compress_hash_arity);
    if (err != eIcicleError::SUCCESS) { return err; }
    Fri verifier_fri = create_fri_ext<scalar_t, extension_t>(
      log_input_size, fri_config.folding_factor, fri_config.stopping_degree, merkle_tree_leaves_hash,
      merkle_tree_compress_hash, 0);
    return verifier_fri.verify(fri_config, fri_transcript_config, fri_proof, valid);
  }
#endif

} // namespace icicle
