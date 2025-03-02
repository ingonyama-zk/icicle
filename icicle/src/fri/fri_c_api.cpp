#include "icicle/fields/field_config.h"
#include "icicle/utils/log.h"
#include "icicle/utils/utils.h"
#include "icicle/fri/fri.h"
#include "icicle/fri/fri_transcript_config.h"
#include <cstddef>

using namespace field_config;

extern "C" {

// Define the FRI handle type
typedef icicle::Fri<scalar_t, scalar_t> FriHandle;

#ifdef EXT_FIELD
typedef icicle::Fri<scalar_t, extension_t> FriHandleExt;
#endif

// Structure to represent the FFI transcript configuration.
struct FriTranscriptConfigFFI {
  Hash* hasher;
  std::byte* domain_separator_label;
  size_t domain_separator_label_len;
  std::byte* round_challenge_label;
  size_t round_challenge_label_len;
  std::byte* commit_label;
  size_t commit_label_len;
  std::byte* nonce_label;
  size_t nonce_label_len;
  std::byte* public_state;
  size_t public_state_len;
  const scalar_t* seed_rng;
};

/**
 * @brief Structure representing creation parameters for the "hash-based" constructor
 *        `create_fri<scalar_t>(folding_factor, stopping_degree, Hash&, output_store_min_layer)`.
 */
struct FriCreateHashFFI {
  size_t input_size;
  size_t folding_factor;
  size_t stopping_degree;
  Hash* merkle_tree_leaves_hash;
  Hash* merkle_tree_compress_hash;
  uint64_t output_store_min_layer;
};

/**
 * @brief Structure representing creation parameters for the "existing Merkle trees" constructor
 *        `create_fri<scalar_t>(folding_factor, stopping_degree, vector<MerkleTree*>&&)`.
 */
struct FriCreateWithTreesFFI {
  size_t folding_factor;
  size_t stopping_degree;
  MerkleTree* merkle_trees;  // An array of MerkleTree* (pointers).
  size_t merkle_trees_count; // Number of items in merkle_trees.
};

/**
 * @brief Creates a new FRI instance from the given FFI transcript configuration
 *        and creation parameters (folding_factor, stopping_degree, hash, etc.).
 * @param create_config Pointer to the creation parameters (FriCreateConfigFFI).
 * @param transcript_config Pointer to the FFI transcript configuration structure.
 * @return Pointer to the created FRI instance (FriHandle*), or nullptr on error.
 */
FriHandle* CONCAT_EXPAND(FIELD, fri_create)(
  const FriCreateHashFFI* create_config, const FriTranscriptConfigFFI* ffi_transcript_config)
{
  if (!create_config || !create_config->merkle_tree_leaves_hash || !create_config->merkle_tree_compress_hash) {
    ICICLE_LOG_ERROR << "Invalid FRI creation config.";
    return nullptr;
  }
  if (!ffi_transcript_config || !ffi_transcript_config->hasher || !ffi_transcript_config->seed_rng) {
    ICICLE_LOG_ERROR << "Invalid FFI transcript configuration for FRI.";
    return nullptr;
  }

  ICICLE_LOG_DEBUG << "Constructing FRI instance from FFI (hash-based)";

  // Convert byte arrays to vectors
  // TODO SHANIE - check if this is the correct way
  std::vector<std::byte> domain_separator_label(
    ffi_transcript_config->domain_separator_label,
    ffi_transcript_config->domain_separator_label + ffi_transcript_config->domain_separator_label_len);
  std::vector<std::byte> round_challenge_label(
    ffi_transcript_config->round_challenge_label,
    ffi_transcript_config->round_challenge_label + ffi_transcript_config->round_challenge_label_len);
  std::vector<std::byte> commit_label(
    ffi_transcript_config->commit_label, ffi_transcript_config->commit_label + ffi_transcript_config->commit_label_len);
  std::vector<std::byte> nonce_label(
    ffi_transcript_config->nonce_label, ffi_transcript_config->nonce_label + ffi_transcript_config->nonce_label_len);
  std::vector<std::byte> public_state(
    ffi_transcript_config->public_state, ffi_transcript_config->public_state + ffi_transcript_config->public_state_len);

  // Construct a FriTranscriptConfig
  FriTranscriptConfig config{
    *(ffi_transcript_config->hasher),
    std::move(domain_separator_label),
    std::move(round_challenge_label),
    std::move(commit_label),
    std::move(nonce_label),
    std::move(public_state),
    *(ffi_transcript_config->seed_rng)};

  // Create and return the Fri instance
  return new icicle::Fri<scalar_t, scalar_t>(icicle::create_fri<scalar_t, scalar_t>(
    create_config->input_size, create_config->folding_factor, create_config->stopping_degree,
    *(create_config->merkle_tree_leaves_hash), *(create_config->merkle_tree_compress_hash),
    create_config->output_store_min_layer));
}

// fri_create_with_trees - Using vector<MerkleTree*>&& constructor

/**
 * @brief Creates a new FRI instance using the vector<MerkleTree*>&& constructor.
 * @param create_config Pointer to a FriCreateWithTreesFFI struct with the necessary parameters.
 * @param transcript_config Pointer to the FFI transcript configuration structure.
 * @return Pointer to the created FRI instance (FriHandle*), or nullptr on error.
 */
FriHandle* CONCAT_EXPAND(FIELD, fri_create_with_trees)(
  const FriCreateWithTreesFFI* create_config, const FriTranscriptConfigFFI* ffi_transcript_config)
{
  if (!create_config || !create_config->merkle_trees) {
    ICICLE_LOG_ERROR << "Invalid FRI creation config with trees.";
    return nullptr;
  }
  if (!ffi_transcript_config || !ffi_transcript_config->hasher || !ffi_transcript_config->seed_rng) {
    ICICLE_LOG_ERROR << "Invalid FFI transcript configuration for FRI.";
    return nullptr;
  }

  ICICLE_LOG_DEBUG << "Constructing FRI instance from FFI (with existing trees)";

  // Convert the raw array of MerkleTree* into a std::vector<MerkleTree*>
  std::vector<MerkleTree> merkle_trees_vec;
  merkle_trees_vec.reserve(create_config->merkle_trees_count);
  for (size_t i = 0; i < create_config->merkle_trees_count; ++i) {
    merkle_trees_vec.push_back(create_config->merkle_trees[i]);
  }

  // Convert byte arrays to vectors
  // TODO SHANIE - check if this is the correct way
  std::vector<std::byte> domain_separator_label(
    ffi_transcript_config->domain_separator_label,
    ffi_transcript_config->domain_separator_label + ffi_transcript_config->domain_separator_label_len);
  std::vector<std::byte> round_challenge_label(
    ffi_transcript_config->round_challenge_label,
    ffi_transcript_config->round_challenge_label + ffi_transcript_config->round_challenge_label_len);
  std::vector<std::byte> commit_label(
    ffi_transcript_config->commit_label, ffi_transcript_config->commit_label + ffi_transcript_config->commit_label_len);
  std::vector<std::byte> nonce_label(
    ffi_transcript_config->nonce_label, ffi_transcript_config->nonce_label + ffi_transcript_config->nonce_label_len);
  std::vector<std::byte> public_state(
    ffi_transcript_config->public_state, ffi_transcript_config->public_state + ffi_transcript_config->public_state_len);

  // Construct a FriTranscriptConfig
  FriTranscriptConfig config{
    *(ffi_transcript_config->hasher),
    std::move(domain_separator_label),
    std::move(round_challenge_label),
    std::move(commit_label),
    std::move(nonce_label),
    std::move(public_state),
    *(ffi_transcript_config->seed_rng)};

  // Create and return the Fri instance
  return new icicle::Fri<scalar_t, scalar_t>(icicle::create_fri<scalar_t, scalar_t>(
    create_config->folding_factor, create_config->stopping_degree, merkle_trees_vec));
}

/**
 * @brief Deletes the given Fri instance.
 * @param fri_handle Pointer to the Fri instance to be deleted.
 * @return eIcicleError indicating the success or failure of the operation.
 */
eIcicleError CONCAT_EXPAND(FIELD, fri_delete)(const FriHandle* fri_handle)
{
  if (!fri_handle) {
    ICICLE_LOG_ERROR << "Cannot delete a null Fri instance.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  ICICLE_LOG_DEBUG << "Destructing Fri instance from FFI";
  delete fri_handle;

  return eIcicleError::SUCCESS;
}

#ifdef EXT_FIELD // EXT_FIELD
FriHandleExt* CONCAT_EXPAND(FIELD, fri_create_ext)(
  const FriCreateHashFFI* create_config, const FriTranscriptConfigFFI* ffi_transcript_config)
{
  if (!create_config || !create_config->merkle_tree_leaves_hash || !create_config->merkle_tree_compress_hash) {
    ICICLE_LOG_ERROR << "Invalid FRI creation config.";
    return nullptr;
  }
  if (!ffi_transcript_config || !ffi_transcript_config->hasher || !ffi_transcript_config->seed_rng) {
    ICICLE_LOG_ERROR << "Invalid FFI transcript configuration for FRI.";
    return nullptr;
  }

  ICICLE_LOG_DEBUG << "Constructing FRI EXT_FIELD instance from FFI (hash-based)";

  // Convert byte arrays to vectors
  std::vector<std::byte> domain_separator_label(
    ffi_transcript_config->domain_separator_label,
    ffi_transcript_config->domain_separator_label + ffi_transcript_config->domain_separator_label_len);
  std::vector<std::byte> round_challenge_label(
    ffi_transcript_config->round_challenge_label,
    ffi_transcript_config->round_challenge_label + ffi_transcript_config->round_challenge_label_len);
  std::vector<std::byte> commit_label(
    ffi_transcript_config->commit_label, ffi_transcript_config->commit_label + ffi_transcript_config->commit_label_len);
  std::vector<std::byte> nonce_label(
    ffi_transcript_config->nonce_label, ffi_transcript_config->nonce_label + ffi_transcript_config->nonce_label_len);
  std::vector<std::byte> public_state(
    ffi_transcript_config->public_state, ffi_transcript_config->public_state + ffi_transcript_config->public_state_len);

  // Construct a FriTranscriptConfig
  FriTranscriptConfig config{
    *(ffi_transcript_config->hasher),
    std::move(domain_separator_label),
    std::move(round_challenge_label),
    std::move(commit_label),
    std::move(nonce_label),
    std::move(public_state),
    *(ffi_transcript_config->seed_rng)};

  // Create and return the Fri instance for the extension field
  return new icicle::Fri<scalar_t, extension_t>(icicle::create_fri<scalar_t, extension_t>(
    create_config->input_size, create_config->folding_factor, create_config->stopping_degree,
    *(create_config->merkle_tree_leaves_hash), *(create_config->merkle_tree_compress_hash),
    create_config->output_store_min_layer));
}

FriHandleExt* CONCAT_EXPAND(FIELD, fri_create_with_trees_ext)(
  const FriCreateWithTreesFFI* create_config, const FriTranscriptConfigFFI* ffi_transcript_config)
{
  if (!create_config || !create_config->merkle_trees) {
    ICICLE_LOG_ERROR << "Invalid FRI creation config with trees.";
    return nullptr;
  }
  if (
    !ffi_transcript_config || !ffi_transcript_config || !ffi_transcript_config->hasher ||
    !ffi_transcript_config->seed_rng) {
    ICICLE_LOG_ERROR << "Invalid FFI transcript configuration for FRI.";
    return nullptr;
  }

  ICICLE_LOG_DEBUG << "Constructing FRI instance from FFI (with existing trees)";

  // Convert the raw array of MerkleTree* into a std::vector<MerkleTree*>
  std::vector<MerkleTree> merkle_trees_vec;
  merkle_trees_vec.reserve(create_config->merkle_trees_count);
  for (size_t i = 0; i < create_config->merkle_trees_count; ++i) {
    merkle_trees_vec.push_back(create_config->merkle_trees[i]);
  }

  // Convert byte arrays to vectors
  // TODO SHANIE - check if this is the correct way
  std::vector<std::byte> domain_separator_label(
    ffi_transcript_config->domain_separator_label,
    ffi_transcript_config->domain_separator_label + ffi_transcript_config->domain_separator_label_len);
  std::vector<std::byte> round_challenge_label(
    ffi_transcript_config->round_challenge_label,
    ffi_transcript_config->round_challenge_label + ffi_transcript_config->round_challenge_label_len);
  std::vector<std::byte> commit_label(
    ffi_transcript_config->commit_label, ffi_transcript_config->commit_label + ffi_transcript_config->commit_label_len);
  std::vector<std::byte> nonce_label(
    ffi_transcript_config->nonce_label, ffi_transcript_config->nonce_label + ffi_transcript_config->nonce_label_len);
  std::vector<std::byte> public_state(
    ffi_transcript_config->public_state, ffi_transcript_config->public_state + ffi_transcript_config->public_state_len);

  // Construct a FriTranscriptConfig
  FriTranscriptConfig config{
    *(ffi_transcript_config->hasher),
    std::move(domain_separator_label),
    std::move(round_challenge_label),
    std::move(commit_label),
    std::move(nonce_label),
    std::move(public_state),
    *(ffi_transcript_config->seed_rng)};

  // Create and return the Fri instance
  return new icicle::Fri<scalar_t, extension_t>(icicle::create_fri<scalar_t, extension_t>(
    create_config->folding_factor, create_config->stopping_degree, merkle_trees_vec));
}

/**
 * @brief Deletes the given Fri instance.
 * @param fri_handle_ext Pointer to the Fri instance to be deleted.
 * @return eIcicleError indicating the success or failure of the operation.
 */
eIcicleError CONCAT_EXPAND(FIELD, fri_delete_ext)(const FriHandleExt* fri_handle_ext)
{
  if (!fri_handle_ext) {
    ICICLE_LOG_ERROR << "Cannot delete a null Fri instance.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  ICICLE_LOG_DEBUG << "Destructing Fri instance from FFI";
  delete fri_handle_ext;

  return eIcicleError::SUCCESS;
}

#endif // EXT_FIELD

} // extern "C"
