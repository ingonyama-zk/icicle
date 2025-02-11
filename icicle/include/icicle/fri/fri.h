#pragma once

#include <memory>
#include <vector>
#include "icicle/errors.h"
#include "icicle/backend/fri_backend.h"
#include "icicle/fri/fri_config.h"
#include "icicle/fri/fri_proof.h"
#include "icicle/fri/fri_transcript_config.h"
#include "icicle/hash/hash.h"
#include "icicle/merkle/merkle_tree.h"

namespace icicle {

/**
 * @brief Forward declaration for the FRI class template.
 */
template <typename F>
class Fri;

/**
 * @brief Constructor for the case where only binary Merkle trees are used 
 *        with a constant hash function.
 *
 * @param input_size The size of the input polynomial - number of evaluations.
 * @param folding_factor The factor by which the codeword is folded each round.
 * @param stopping_degree The minimal polynomial degree at which to stop folding.
 * @param hash_for_merkle_tree The hash function used for the Merkle commitments.
 * @param output_store_min_layer (Optional) The layer at which to store partial results. Default = 0.
 * @return A `Fri<F>` object built around the chosen backend.
 */
template <typename F>
Fri<F> create_fri(
    size_t input_size,
    size_t folding_factor,
    size_t stopping_degree,
    Hash& hash_for_merkle_tree,
    uint64_t output_store_min_layer = 0);

/**
 * @brief Constructor for the case where Merkle trees are already given.
 *
 * @param folding_factor The factor by which the codeword is folded each round.
 * @param stopping_degree The minimal polynomial degree at which to stop folding.
 * @param merkle_trees A moved vector of `MerkleTree` objects.
 * @return A `Fri<F>` object built around the chosen backend.
 */
template <typename F>
Fri<F> create_fri(
    size_t folding_factor,
    size_t stopping_degree,
    std::vector<MerkleTree>&& merkle_trees);

/**
 * @brief Class for performing FRI operations.
 *
 * This class provides a high-level interface for constructing and managing a FRI proof.
 *
 * @tparam F The field type used in the FRI protocol.
 */
template <typename F>
class Fri
{
public:
    /**
     * @brief Constructor for the Fri class.
     * @param backend A shared pointer to the backend (FriBackend<F>) responsible for FRI operations.
     */
    explicit Fri(std::shared_ptr<FriBackend<F>> backend)
        : m_backend(std::move(backend))
    {}

    /**
     * @brief Generate a FRI proof from the given polynomial evaluations (or input data).
     * @param fri_config Configuration for FRI operations (e.g., proof-of-work, queries).
     * @param fri_transcript_config Configuration for encoding/hashing (Fiat-Shamir).
     * @param input_data Evaluations or other relevant data for constructing the proof.
     * @param fri_proof Reference to a FriProof object (output).
     * @return An eIcicleError indicating success or failure.
     */
    eIcicleError get_fri_proof(
        const FriConfig& fri_config,
        const FriTranscriptConfig<F>& fri_transcript_config,
        const std::vector<F*>& input_data,
        FriProof<F>& fri_proof /* OUT */) const
    {
        return m_backend->get_fri_proof(fri_config, fri_transcript_config, input_data, fri_proof);
    }

    /**
     * @brief Verify a FRI proof.
     * @param fri_config Configuration for FRI operations.
     * @param fri_transcript_config Configuration for encoding/hashing (Fiat-Shamir).
     * @param fri_proof The proof object to verify.
     * @param verification_pass (OUT) Set to true if verification succeeds, false otherwise.
     * @return An eIcicleError indicating success or failure.
     */
    eIcicleError verify(
        const FriConfig& fri_config,
        const FriTranscriptConfig<F>& fri_transcript_config,
        const FriProof<F>& fri_proof,
        bool& verification_pass /* OUT */) const
    {
        return eIcicleError::API_NOT_IMPLEMENTED;
    }

private:
    std::shared_ptr<FriBackend<F>> m_backend; // Shared pointer to the backend for FRI operations.
};

} // namespace icicle
