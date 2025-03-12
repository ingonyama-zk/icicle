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
   * @brief Generates a FRI proof using a binary Merkle tree structure.
   *
   * This function constructs a FRI proof by applying the Fast Reed-Solomon
   * Interactive Oracle Proof of Proximity (FRI) protocol. The proof is built
   * using a Merkle tree with a predefined hash function.
   *
   * @param fri_config Configuration parameters for the FRI protocol.
   * @param fri_transcript_config Configuration for the Fiat-Shamir transcript used in FRI.
   * @param input_data Pointer to the polynomial evaluations.
   * @param input_size The number of evaluations in the input polynomial.
   * @param merkle_tree_leaves_hash The hash function used for Merkle tree leaves.
   * @param merkle_tree_compress_hash The hash function used for compressing Merkle tree nodes.
   * @param output_store_min_layer The layer at which to store partial results. Default = 0.
   * @param fri_proof (OUT) The generated FRI proof.
   * @return `eIcicleError` indicating success or failure of the proof generation.
   */

  template <typename S, typename F>
  eIcicleError get_fri_proof_merkle_tree(
    const FriConfig& fri_config,
    const FriTranscriptConfig<F>& fri_transcript_config,
    const F* input_data,
    const size_t input_size,
    Hash merkle_tree_leaves_hash,
    Hash merkle_tree_compress_hash,
    const uint64_t output_store_min_layer,
    FriProof<F>& fri_proof /* OUT */);

  /**
   * @brief Verifies a given FRI proof using a binary Merkle tree structure.
   *
   * This function checks the validity of a FRI proof by reconstructing the
   * Merkle tree and ensuring consistency with the committed data. The verification
   * process leverages the Fiat-Shamir heuristic to maintain non-interactivity.
   *
   * @param fri_config Configuration parameters for the FRI protocol.
   * @param fri_transcript_config Configuration for the Fiat-Shamir transcript used in FRI.
   * @param fri_proof The FRI proof to be verified.
   * @param merkle_tree_leaves_hash The hash function used for Merkle tree leaves.
   * @param merkle_tree_compress_hash The hash function used for compressing Merkle tree nodes.
   * @param valid (OUT) Boolean flag indicating whether the proof is valid.
   * @return `eIcicleError` indicating success or failure of the verification process.
   */

  template <typename S, typename F>
  eIcicleError verify_fri_merkle_tree(
    const FriConfig& fri_config,
    const FriTranscriptConfig<F>& fri_transcript_config,
    const FriProof<F>& fri_proof,
    Hash merkle_tree_leaves_hash,
    Hash merkle_tree_compress_hash,
    bool& valid /* OUT */);

  namespace fri_merkle_tree {
    template <typename S, typename F>
    inline static eIcicleError prove(
      const FriConfig& fri_config,
      const FriTranscriptConfig<F>& fri_transcript_config,
      const F* input_data,
      const size_t input_size,
      Hash merkle_tree_leaves_hash,
      Hash merkle_tree_compress_hash,
      const uint64_t output_store_min_layer,
      FriProof<F>& fri_proof /* OUT */)
    {
      return get_fri_proof_merkle_tree<S, F>(
        fri_config, fri_transcript_config, input_data, input_size, merkle_tree_leaves_hash, merkle_tree_compress_hash,
        output_store_min_layer, fri_proof /* OUT */);
    }

    template <typename S, typename F>
    inline static eIcicleError verify(
      const FriConfig& fri_config,
      const FriTranscriptConfig<F>& fri_transcript_config,
      const FriProof<F>& fri_proof,
      Hash merkle_tree_leaves_hash,
      Hash merkle_tree_compress_hash,
      bool& valid /* OUT */)
    {
      return verify_fri_merkle_tree<S, F>(
        fri_config, fri_transcript_config, fri_proof, merkle_tree_leaves_hash, merkle_tree_compress_hash,
        valid /* OUT */);
    }
  }; // namespace fri_merkle_tree

} // namespace icicle
