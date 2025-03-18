#include <memory>
#include "icicle/fri/fri.h"
#include "icicle/fri/fri_transcript_config.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"

using namespace field_config;
using namespace icicle;

template <typename T>
struct FFIFriTranscriptConfig {
  Hash* hasher_handle;
  uint8_t* domain_separator_label;
  size_t domain_separator_label_len;

  uint8_t* round_challenge_label;
  size_t round_challenge_label_len;

  uint8_t* commit_phase_label;
  size_t commit_phase_label_len;

  uint8_t* nonce_label;
  size_t nonce_label_len;

  uint8_t* public_state;
  size_t public_state_len;

  const T* seed_rng;
};

template <typename T>
FriTranscriptConfig<T> convert_ffi_transcript_config(const FFIFriTranscriptConfig<T>* ffi_transcript_config)
{
  // Convert byte arrays to vectors
  std::vector<std::byte> domain_separator_label(
    reinterpret_cast<const std::byte*>(ffi_transcript_config->domain_separator_label),
    reinterpret_cast<const std::byte*>(ffi_transcript_config->domain_separator_label) +
      ffi_transcript_config->domain_separator_label_len);

  std::vector<std::byte> round_challenge_label(
    reinterpret_cast<const std::byte*>(ffi_transcript_config->round_challenge_label),
    reinterpret_cast<const std::byte*>(ffi_transcript_config->round_challenge_label) +
      ffi_transcript_config->round_challenge_label_len);

  std::vector<std::byte> commit_phase_label(
    reinterpret_cast<const std::byte*>(ffi_transcript_config->commit_phase_label),
    reinterpret_cast<const std::byte*>(ffi_transcript_config->commit_phase_label) +
      ffi_transcript_config->commit_phase_label_len);

  std::vector<std::byte> nonce_label(
    reinterpret_cast<const std::byte*>(ffi_transcript_config->nonce_label),
    reinterpret_cast<const std::byte*>(ffi_transcript_config->nonce_label) + ffi_transcript_config->nonce_label_len);

  std::vector<std::byte> public_state(
    reinterpret_cast<const std::byte*>(ffi_transcript_config->public_state),
    reinterpret_cast<const std::byte*>(ffi_transcript_config->public_state) + ffi_transcript_config->public_state_len);

  // Construct and return internal config
  FriTranscriptConfig transcript_config{
    *ffi_transcript_config->hasher_handle,
    std::move(domain_separator_label),
    std::move(round_challenge_label),
    std::move(commit_phase_label),
    std::move(nonce_label),
    std::move(public_state),
    *ffi_transcript_config->seed_rng};

  return transcript_config;
}

extern "C" {
typedef Hash* HasherHandle;

// FriProof API

typedef FriProof<scalar_t>* FriProofHandle;

FriProofHandle CONCAT_EXPAND(ICICLE_FFI_PREFIX, icicle_initialize_fri_proof)() { return new FriProof<scalar_t>(); }

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, icicle_delete_fri_proof)(FriProofHandle proof_ptr)
{
  if (!proof_ptr) return eIcicleError::INVALID_POINTER;
  delete proof_ptr;
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_get_proof_sizes)(
  FriProofHandle proof_ptr, size_t& nof_queries, size_t& nof_rounds)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "CANNOT USE null proof_ptr";
    return eIcicleError::INVALID_POINTER;
  }
  nof_queries = proof_ptr->get_nof_fri_queries();
  nof_rounds = proof_ptr->get_nof_fri_rounds();
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_get_round_proof_at)(
  FriProofHandle proof_ptr, size_t query_idx, MerkleProof** proofs)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "CANNOT USE null proof_ptr";
    return eIcicleError::INVALID_POINTER;
  }
  if (query_idx >= proof_ptr->get_nof_fri_queries()) {
    ICICLE_LOG_ERROR << "query_idx IS OUT OF RANGE";
    return eIcicleError::INVALID_ARGUMENT;
  }
  *proofs = proof_ptr->get_proofs_at_query(query_idx);
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_get_final_poly_size)(FriProofHandle proof_ptr, size_t& result)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "CANNOT USE null proof_ptr";
    return eIcicleError::INVALID_POINTER;
  }
  result = proof_ptr->get_final_poly_size();
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_get_final_poly)(FriProofHandle proof_ptr, scalar_t** final_poly)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "CANNOT USE null proof_ptr";
    return eIcicleError::INVALID_POINTER;
  }
  *final_poly = proof_ptr->get_final_poly();
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_get_pow_nonce)(FriProofHandle proof_ptr, uint64_t& result)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "CANNOT USE null proof_ptr";
    return eIcicleError::INVALID_POINTER;
  }
  result = proof_ptr->get_pow_nonce();
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, get_fri_proof_mt)(
  const FriConfig& fri_config,
  const FFIFriTranscriptConfig<scalar_t>* ffi_transcript_config,
  const scalar_t* input_data,
  const size_t input_size,
  Hash& merkle_tree_leaves_hash,
  Hash& merkle_tree_compress_hash,
  const uint64_t output_store_min_layer,
  FriProof<scalar_t>& fri_proof /* OUT */)
{
  if (!ffi_transcript_config || !ffi_transcript_config->hasher_handle || !ffi_transcript_config->seed_rng) {
    ICICLE_LOG_ERROR << "Invalid FFI Fri transcript configuration.";
    return eIcicleError::INVALID_POINTER;
  }
  auto fri_transcript_config = convert_ffi_transcript_config(ffi_transcript_config);
  return prove_fri_merkle_tree<scalar_t>(
    fri_config, fri_transcript_config, input_data, input_size, merkle_tree_leaves_hash, merkle_tree_compress_hash,
    output_store_min_layer, fri_proof);
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, verify_fri_mt)(
  const FriConfig& fri_config,
  const FFIFriTranscriptConfig<scalar_t>* ffi_transcript_config,
  FriProof<scalar_t>& fri_proof,
  Hash& merkle_tree_leaves_hash,
  Hash& merkle_tree_compress_hash,
  bool& valid /* OUT */)
{
  if (!ffi_transcript_config || !ffi_transcript_config->hasher_handle || !ffi_transcript_config->seed_rng) {
    ICICLE_LOG_ERROR << "Invalid FFI Fri transcript configuration.";
    return eIcicleError::INVALID_POINTER;
  }
  auto fri_transcript_config = convert_ffi_transcript_config(ffi_transcript_config);
  return verify_fri_merkle_tree<scalar_t>(
    fri_config, fri_transcript_config, fri_proof, merkle_tree_leaves_hash, merkle_tree_compress_hash, valid);
}

#ifdef EXT_FIELD

typedef FriTranscriptConfig<extension_t>* FriTranscriptConfigExtensionHandle;
typedef FriProof<extension_t>* FriProofExtensionHandle;

FriProofExtensionHandle CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_icicle_initialize_fri_proof)()
{
  return new FriProof<extension_t>();
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_icicle_delete_fri_proof)(FriProofExtensionHandle proof_ptr)
{
  if (!proof_ptr) return eIcicleError::INVALID_POINTER;
  delete proof_ptr;
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_get_proof_sizes)(
  FriProofExtensionHandle proof_ptr, size_t& nof_queries, size_t& nof_rounds)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "CANNOT USE null proof_ptr";
    return eIcicleError::INVALID_POINTER;
  }
  nof_queries = proof_ptr->get_nof_fri_queries();
  nof_rounds = proof_ptr->get_nof_fri_rounds();
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_get_round_proof_at)(
  FriProofExtensionHandle proof_ptr, size_t query_idx, MerkleProof** proofs)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "CANNOT USE null proof_ptr";
    return eIcicleError::INVALID_POINTER;
  }
  if (query_idx >= proof_ptr->get_nof_fri_queries()) {
    ICICLE_LOG_ERROR << "query_idx IS OUT OF RANGE";
    return eIcicleError::INVALID_ARGUMENT;
  }
  *proofs = proof_ptr->get_proofs_at_query(query_idx);
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_get_final_poly_size)(
  FriProofExtensionHandle proof_ptr, size_t& result)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "CANNOT USE null proof_ptr";
    return eIcicleError::INVALID_POINTER;
  }
  result = proof_ptr->get_final_poly_size();
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_get_final_poly)(
  FriProofExtensionHandle proof_ptr, extension_t** final_poly)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "CANNOT USE null proof_ptr";
    return eIcicleError::INVALID_POINTER;
  }
  *final_poly = proof_ptr->get_final_poly();
  return eIcicleError::SUCCESS;
}

eIcicleError
CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_get_pow_nonce)(FriProofExtensionHandle proof_ptr, uint64_t& result)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "CANNOT USE null proof_ptr";
    return eIcicleError::INVALID_POINTER;
  }
  result = proof_ptr->get_pow_nonce();
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_get_fri_proof_mt)(
  const FriConfig& fri_config,
  const FFIFriTranscriptConfig<extension_t>* ffi_transcript_config,
  const extension_t* input_data,
  const size_t input_size,
  Hash& merkle_tree_leaves_hash,
  Hash& merkle_tree_compress_hash,
  const uint64_t output_store_min_layer,
  FriProof<extension_t>& fri_proof /* OUT */)
{
  if (!ffi_transcript_config || !ffi_transcript_config->hasher_handle || !ffi_transcript_config->seed_rng) {
    ICICLE_LOG_ERROR << "Invalid FFI Fri transcript configuration.";
    return eIcicleError::INVALID_POINTER;
  }
  auto fri_transcript_config = convert_ffi_transcript_config(ffi_transcript_config);
  return prove_fri_merkle_tree<extension_t>(
    fri_config, fri_transcript_config, input_data, input_size, merkle_tree_leaves_hash, merkle_tree_compress_hash,
    output_store_min_layer, fri_proof);
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_verify_fri_mt)(
  const FriConfig& fri_config,
  const FFIFriTranscriptConfig<extension_t>* ffi_transcript_config,
  FriProof<extension_t>& fri_proof,
  Hash& merkle_tree_leaves_hash,
  Hash& merkle_tree_compress_hash,
  bool& valid /* OUT */)
{
  if (!ffi_transcript_config || !ffi_transcript_config->hasher_handle || !ffi_transcript_config->seed_rng) {
    ICICLE_LOG_ERROR << "Invalid FFI Fri transcript configuration.";
    return eIcicleError::INVALID_POINTER;
  }
  auto fri_transcript_config = convert_ffi_transcript_config(ffi_transcript_config);
  return verify_fri_merkle_tree<extension_t>(
    fri_config, fri_transcript_config, fri_proof, merkle_tree_leaves_hash, merkle_tree_compress_hash, valid);
}
#endif
}