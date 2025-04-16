#include <memory>
#include "icicle/fri/fri.h"
#include "icicle/fri/fri_transcript_config.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"

using namespace field_config;
using namespace icicle;

typedef Hash* HasherHandle;

template <typename T>
struct FFIFriTranscriptConfig {
  HasherHandle hasher_handle;
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

// FriProof API

typedef FriProof<scalar_t>* FriProofHandle;

FriProofHandle CONCAT_EXPAND(ICICLE_FFI_PREFIX, icicle_initialize_fri_proof)() { return new FriProof<scalar_t>(); }

FriProofHandle CONCAT_EXPAND(ICICLE_FFI_PREFIX, icicle_create_with_arguments_fri_proof)(
  MerkleProof*** query_proofs_ffi,
  size_t nof_queries,
  size_t nof_rounds,
  scalar_t* final_poly_ffi,
  size_t final_poly_size,
  uint64_t pow_nonce)
{
  std::vector<std::vector<MerkleProof>> query_proofs(nof_queries);
  for (auto i = 0; i < nof_queries; ++i) {
    std::vector<MerkleProof> proofs_per_query(nof_rounds);
    for (auto j = 0; j < nof_rounds; ++j) {
      proofs_per_query[j] = *query_proofs_ffi[i][j];
    }
    query_proofs[i] = proofs_per_query;
  }
  std::vector<scalar_t> final_poly = {final_poly_ffi, final_poly_ffi + final_poly_size};
  return new FriProof<scalar_t>(std::move(query_proofs), std::move(final_poly), pow_nonce);
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, icicle_delete_fri_proof)(FriProofHandle proof_ptr)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null - cannot delete";
    return eIcicleError::INVALID_POINTER;
  }
  delete proof_ptr;
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_get_nof_queries)(FriProofHandle proof_ptr, size_t* nof_queries)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  if (!nof_queries) {
    ICICLE_LOG_ERROR << "nof_queries is null — cannot set result";
    return eIcicleError::INVALID_POINTER;
  }

  *nof_queries = proof_ptr->get_nof_fri_queries();
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_get_nof_rounds)(FriProofHandle proof_ptr, size_t* nof_rounds)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  if (!nof_rounds) {
    ICICLE_LOG_ERROR << "nof_rounds is null — cannot set result";
    return eIcicleError::INVALID_POINTER;
  }

  *nof_rounds = proof_ptr->get_nof_fri_rounds();
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_get_round_proofs_for_query)(
  FriProofHandle proof_ptr, size_t query_idx, MerkleProof** proofs)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  if (!proofs) {
    ICICLE_LOG_ERROR << "proofs is null — cannot set result";
    return eIcicleError::INVALID_POINTER;
  }
  size_t nof_queries = proof_ptr->get_nof_fri_queries();
  if (query_idx >= nof_queries) {
    ICICLE_LOG_ERROR << "query_idx (" << query_idx << ") out of range (max " << nof_queries - 1 << ")";
    return eIcicleError::INVALID_ARGUMENT;
  }
  for (auto i = 0; i < proof_ptr->get_nof_fri_rounds(); ++i) {
    proofs[i] = &proof_ptr->get_query_proof_slot(query_idx, i);
  }
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_get_final_poly_size)(FriProofHandle proof_ptr, size_t* result)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  if (!result) {
    ICICLE_LOG_ERROR << "result is null — cannot set result";
    return eIcicleError::INVALID_POINTER;
  }
  *result = proof_ptr->get_final_poly_size();
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_get_final_poly)(FriProofHandle proof_ptr, scalar_t** final_poly)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  *final_poly = proof_ptr->get_final_poly();
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_get_pow_nonce)(FriProofHandle proof_ptr, uint64_t* result)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  if (!result) {
    ICICLE_LOG_ERROR << "result is null — cannot set result";
    return eIcicleError::INVALID_POINTER;
  }
  *result = proof_ptr->get_pow_nonce();
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_get_serialized_size)(FriProofHandle proof_ptr, size_t* result)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  return proof_ptr->serialized_size(*result);
}

eIcicleError
CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_serialize)(FriProofHandle proof_ptr, std::byte* buffer, size_t size)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  if (!buffer) {
    ICICLE_LOG_ERROR << "buffer is null — cannot serialize Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  size_t expected_size = 0;
  eIcicleError err = proof_ptr->serialized_size(expected_size);
  if (err != eIcicleError::SUCCESS) {
    ICICLE_LOG_ERROR << "Cannot get serialized size of Fri proof";
    return err;
  }
  if (size < expected_size) {
    ICICLE_LOG_ERROR << "buffer is too small — cannot serialize Fri proof";
    return eIcicleError::INVALID_ARGUMENT;
  }
  return proof_ptr->serialize(buffer);
}

eIcicleError
CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_deserialize)(FriProofHandle* proof_ptr, std::byte* buffer, size_t size)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot deserialize Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  if (!buffer) {
    ICICLE_LOG_ERROR << "buffer is null — cannot deserialize Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  *proof_ptr = new FriProof<scalar_t>();
  return (*proof_ptr)->deserialize(buffer, size);
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_serialize_to_file)(
  FriProofHandle proof_ptr, const char* filename, size_t filename_len)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  if (!filename || !filename_len) {
    ICICLE_LOG_ERROR << "filename is null or filename_len is 0 — cannot serialize Fri proof to file";
    return eIcicleError::INVALID_ARGUMENT;
  }
  std::string filename_str(filename, filename_len);
  return proof_ptr->serialize_to_file(filename_str);
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_deserialize_from_file)(
  FriProofHandle* proof_ptr, const char* filename, size_t filename_len)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot deserialize Fri proof from file";
    return eIcicleError::INVALID_POINTER;
  }
  if (!filename || !filename_len) {
    ICICLE_LOG_ERROR << "filename is null or filename_len is 0 — cannot deserialize Fri proof from file";
    return eIcicleError::INVALID_ARGUMENT;
  }
  std::string filename_str(filename, filename_len);
  *proof_ptr = new FriProof<scalar_t>();
  return (*proof_ptr)->deserialize_from_file(filename_str);
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_merkle_tree_prove)(
  const FriConfig* fri_config,
  const FFIFriTranscriptConfig<scalar_t>* ffi_transcript_config,
  const scalar_t* input_data,
  const size_t input_size,
  HasherHandle merkle_tree_leaves_hash,
  HasherHandle merkle_tree_compress_hash,
  const uint64_t output_store_min_layer,
  FriProofHandle fri_proof /* OUT */)
{
  if (!fri_config) {
    ICICLE_LOG_ERROR << "fri_config is null — cannot retrieve Fri config";
    return eIcicleError::INVALID_POINTER;
  }
  if (!ffi_transcript_config || !ffi_transcript_config->hasher_handle || !ffi_transcript_config->seed_rng) {
    ICICLE_LOG_ERROR << "Invalid FFI Fri transcript configuration.";
    return eIcicleError::INVALID_POINTER;
  }
  if (!input_data) {
    ICICLE_LOG_ERROR << "input_data is null — cannot retrieve Input data";
    return eIcicleError::INVALID_POINTER;
  }
  if (!merkle_tree_leaves_hash) {
    ICICLE_LOG_ERROR << "merkle_tree_leaves_hash is null — cannot retrieve Merkle tree leaves hash";
  }
  if (!merkle_tree_compress_hash) {
    ICICLE_LOG_ERROR << "merkle_tree_compress_hash is null — cannot retrieve Merkle tree compress hash";
  }
  if (!fri_proof) {
    ICICLE_LOG_ERROR << "fri_proof is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  auto fri_transcript_config = convert_ffi_transcript_config(ffi_transcript_config);
  return prove_fri_merkle_tree<scalar_t>(
    *fri_config, fri_transcript_config, input_data, input_size, *merkle_tree_leaves_hash, *merkle_tree_compress_hash,
    output_store_min_layer, *fri_proof);
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_merkle_tree_verify)(
  const FriConfig* fri_config,
  const FFIFriTranscriptConfig<scalar_t>* ffi_transcript_config,
  FriProofHandle fri_proof,
  HasherHandle merkle_tree_leaves_hash,
  HasherHandle merkle_tree_compress_hash,
  bool* valid /* OUT */)
{
  if (!fri_config) {
    ICICLE_LOG_ERROR << "fri_config is null — cannot retrieve Fri config";
    return eIcicleError::INVALID_POINTER;
  }
  if (!ffi_transcript_config || !ffi_transcript_config->hasher_handle || !ffi_transcript_config->seed_rng) {
    ICICLE_LOG_ERROR << "Invalid FFI Fri transcript configuration.";
    return eIcicleError::INVALID_POINTER;
  }
  if (!merkle_tree_leaves_hash) {
    ICICLE_LOG_ERROR << "merkle_tree_leaves_hash is null — cannot retrieve Merkle tree leaves hash";
  }
  if (!merkle_tree_compress_hash) {
    ICICLE_LOG_ERROR << "merkle_tree_compress_hash is null — cannot retrieve Merkle tree compress hash";
  }
  if (!fri_proof) {
    ICICLE_LOG_ERROR << "fri_proof is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  if (!valid) {
    ICICLE_LOG_ERROR << "valid is null — cannot set result";
    return eIcicleError::INVALID_POINTER;
  }

  auto fri_transcript_config = convert_ffi_transcript_config(ffi_transcript_config);
  return verify_fri_merkle_tree<scalar_t>(
    *fri_config, fri_transcript_config, *fri_proof, *merkle_tree_leaves_hash, *merkle_tree_compress_hash, *valid);
}

#ifdef EXT_FIELD

typedef FriTranscriptConfig<extension_t>* FriTranscriptConfigExtensionHandle;
typedef FriProof<extension_t>* FriProofExtensionHandle;

FriProofExtensionHandle CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_icicle_initialize_fri_proof)()
{
  return new FriProof<extension_t>();
}

FriProofExtensionHandle CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_icicle_create_with_arguments_fri_proof)(
  MerkleProof*** query_proofs_ffi,
  size_t nof_queries,
  size_t nof_rounds,
  extension_t* final_poly_ffi,
  size_t final_poly_size,
  uint64_t pow_nonce)
{
  std::vector<std::vector<MerkleProof>> query_proofs(nof_queries);
  for (auto i = 0; i < nof_queries; ++i) {
    std::vector<MerkleProof> proofs_per_query(nof_rounds);
    for (auto j = 0; j < nof_rounds; ++j) {
      proofs_per_query[j] = *query_proofs_ffi[i][j];
    }
    query_proofs[i] = proofs_per_query;
  }
  std::vector<extension_t> final_poly = {final_poly_ffi, final_poly_ffi + final_poly_size};
  return new FriProof<extension_t>(query_proofs, final_poly, pow_nonce);
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_icicle_delete_fri_proof)(FriProofExtensionHandle proof_ptr)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null - cannot delete";
    return eIcicleError::INVALID_POINTER;
  }
  delete proof_ptr;
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_get_nof_queries)(
  FriProofExtensionHandle proof_ptr, size_t* nof_queries)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  if (!nof_queries) {
    ICICLE_LOG_ERROR << "nof_queries is null — cannot set result";
    return eIcicleError::INVALID_POINTER;
  }

  *nof_queries = proof_ptr->get_nof_fri_queries();
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_get_nof_rounds)(
  FriProofExtensionHandle proof_ptr, size_t* nof_rounds)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  if (!nof_rounds) {
    ICICLE_LOG_ERROR << "nof_rounds is null — cannot set result";
    return eIcicleError::INVALID_POINTER;
  }

  *nof_rounds = proof_ptr->get_nof_fri_rounds();
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_get_round_proofs_for_query)(
  FriProofExtensionHandle proof_ptr, size_t query_idx, MerkleProof** proofs)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  if (!proofs) {
    ICICLE_LOG_ERROR << "proofs is null — cannot set result";
    return eIcicleError::INVALID_POINTER;
  }
  size_t nof_queries = proof_ptr->get_nof_fri_queries();
  if (query_idx >= nof_queries) {
    ICICLE_LOG_ERROR << "query_idx (" << query_idx << ") out of range (max " << nof_queries - 1 << ")";
    return eIcicleError::INVALID_ARGUMENT;
  }
  for (auto i = 0; i < proof_ptr->get_nof_fri_rounds(); ++i) {
    proofs[i] = &proof_ptr->get_query_proof_slot(query_idx, i);
  }
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_get_final_poly_size)(
  FriProofExtensionHandle proof_ptr, size_t* result)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  if (!result) {
    ICICLE_LOG_ERROR << "result is null — cannot set result";
    return eIcicleError::INVALID_POINTER;
  }
  *result = proof_ptr->get_final_poly_size();
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_get_final_poly)(
  FriProofExtensionHandle proof_ptr, extension_t** final_poly)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  if (!final_poly) {
    ICICLE_LOG_ERROR << "final_poly is null — cannot set result";
    return eIcicleError::INVALID_POINTER;
  }
  *final_poly = proof_ptr->get_final_poly();
  return eIcicleError::SUCCESS;
}

eIcicleError
CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_get_pow_nonce)(FriProofExtensionHandle proof_ptr, uint64_t* result)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot retrieve Merkle proof";
    return eIcicleError::INVALID_POINTER;
  }
  if (!result) {
    ICICLE_LOG_ERROR << "result is null — cannot set result";
    return eIcicleError::INVALID_POINTER;
  }
  *result = proof_ptr->get_pow_nonce();
  return eIcicleError::SUCCESS;
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_get_serialized_size)(
  FriProofExtensionHandle proof_ptr, size_t* result)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  return proof_ptr->serialized_size(*result);
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_serialize)(
  FriProofExtensionHandle proof_ptr, std::byte* buffer, size_t size)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  if (!buffer) {
    ICICLE_LOG_ERROR << "buffer is null — cannot serialize Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  size_t expected_size = 0;
  eIcicleError err = proof_ptr->serialized_size(expected_size);
  if (err != eIcicleError::SUCCESS) {
    ICICLE_LOG_ERROR << "Cannot get serialized size of Fri proof";
    return err;
  }
  if (size < expected_size) {
    ICICLE_LOG_ERROR << "buffer is too small — cannot serialize Fri proof";
    return eIcicleError::INVALID_ARGUMENT;
  }
  return proof_ptr->serialize(buffer);
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_deserialize)(
  FriProofExtensionHandle* proof_ptr, std::byte* buffer, size_t size)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot deserialize Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  if (!buffer) {
    ICICLE_LOG_ERROR << "buffer is null — cannot deserialize Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  *proof_ptr = new FriProof<extension_t>();
  return (*proof_ptr)->deserialize(buffer, size);
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_serialize_to_file)(
  FriProofExtensionHandle proof_ptr, const char* filename, size_t filename_len)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  if (!filename || !filename_len) {
    ICICLE_LOG_ERROR << "filename is null or filename_len is 0 — cannot serialize Fri proof to file";
    return eIcicleError::INVALID_ARGUMENT;
  }
  std::string filename_str(filename, filename_len);
  return proof_ptr->serialize_to_file(filename_str);
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_deserialize_from_file)(
  FriProofExtensionHandle* proof_ptr, const char* filename, size_t filename_len)
{
  if (!proof_ptr) {
    ICICLE_LOG_ERROR << "proof_ptr is null — cannot deserialize Fri proof from file";
    return eIcicleError::INVALID_POINTER;
  }
  if (!filename || !filename_len) {
    ICICLE_LOG_ERROR << "filename is null or filename_len is 0 — cannot deserialize Fri proof from file";
    return eIcicleError::INVALID_ARGUMENT;
  }

  std::string filename_str(filename, filename_len);
  *proof_ptr = new FriProof<extension_t>();
  return (*proof_ptr)->deserialize_from_file(filename_str);
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_merkle_tree_prove)(
  const FriConfig* fri_config,
  const FFIFriTranscriptConfig<extension_t>* ffi_transcript_config,
  const extension_t* input_data,
  const size_t input_size,
  HasherHandle merkle_tree_leaves_hash,
  HasherHandle merkle_tree_compress_hash,
  const uint64_t output_store_min_layer,
  FriProofExtensionHandle fri_proof /* OUT */)
{
  if (!fri_config) {
    ICICLE_LOG_ERROR << "fri_config is null — cannot retrieve Fri config";
    return eIcicleError::INVALID_POINTER;
  }
  if (!ffi_transcript_config || !ffi_transcript_config->hasher_handle || !ffi_transcript_config->seed_rng) {
    ICICLE_LOG_ERROR << "Invalid FFI Fri transcript configuration.";
    return eIcicleError::INVALID_POINTER;
  }
  if (!input_data) {
    ICICLE_LOG_ERROR << "input_data is null — cannot retrieve Input data";
    return eIcicleError::INVALID_POINTER;
  }
  if (!merkle_tree_leaves_hash) {
    ICICLE_LOG_ERROR << "merkle_tree_leaves_hash is null — cannot retrieve Merkle tree leaves hash";
  }
  if (!merkle_tree_compress_hash) {
    ICICLE_LOG_ERROR << "merkle_tree_compress_hash is null — cannot retrieve Merkle tree compress hash";
  }
  if (!fri_proof) {
    ICICLE_LOG_ERROR << "fri_proof is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  auto fri_transcript_config = convert_ffi_transcript_config(ffi_transcript_config);
  return prove_fri_merkle_tree<extension_t>(
    *fri_config, fri_transcript_config, input_data, input_size, *merkle_tree_leaves_hash, *merkle_tree_compress_hash,
    output_store_min_layer, *fri_proof);
}

eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_merkle_tree_verify)(
  const FriConfig* fri_config,
  const FFIFriTranscriptConfig<extension_t>* ffi_transcript_config,
  FriProofExtensionHandle fri_proof,
  HasherHandle merkle_tree_leaves_hash,
  HasherHandle merkle_tree_compress_hash,
  bool* valid /* OUT */)
{
  if (!fri_config) {
    ICICLE_LOG_ERROR << "fri_config is null — cannot retrieve Fri config";
    return eIcicleError::INVALID_POINTER;
  }
  if (!ffi_transcript_config || !ffi_transcript_config->hasher_handle || !ffi_transcript_config->seed_rng) {
    ICICLE_LOG_ERROR << "Invalid FFI Fri transcript configuration.";
    return eIcicleError::INVALID_POINTER;
  }
  if (!merkle_tree_leaves_hash) {
    ICICLE_LOG_ERROR << "merkle_tree_leaves_hash is null — cannot retrieve Merkle tree leaves hash";
  }
  if (!merkle_tree_compress_hash) {
    ICICLE_LOG_ERROR << "merkle_tree_compress_hash is null — cannot retrieve Merkle tree compress hash";
  }
  if (!fri_proof) {
    ICICLE_LOG_ERROR << "fri_proof is null — cannot retrieve Fri proof";
    return eIcicleError::INVALID_POINTER;
  }
  if (!valid) {
    ICICLE_LOG_ERROR << "valid is null — cannot set result";
    return eIcicleError::INVALID_POINTER;
  }
  auto fri_transcript_config = convert_ffi_transcript_config(ffi_transcript_config);
  return verify_fri_merkle_tree<extension_t>(
    *fri_config, fri_transcript_config, *fri_proof, *merkle_tree_leaves_hash, *merkle_tree_compress_hash, *valid);
}
#endif
}