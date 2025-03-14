#include <memory>
#include "icicle/fri/fri.h"
#include "icicle/fri/fri_transcript_config.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"

using namespace field_config;
using namespace icicle;

extern "C" {
  typedef Hash* HasherHandle;

  // FriProof API

  typedef FriProof<scalar_t>* FriProofHandle;

  FriProofHandle CONCAT_EXPAND(FIELD, icicle_initialize_fri_proof)() {
    return new FriProof<scalar_t>();
  }

  eIcicleError icicle_delete_fri_proof(FriProofHandle proof_ptr) {
    if (!proof_ptr) return eIcicleError::INVALID_POINTER;
    delete proof_ptr;
    return eIcicleError::SUCCESS;
  }

  // FriTranscriptConfig API

  typedef FriTranscriptConfig<scalar_t>* FriTranscriptConfigHandle;

  FriTranscriptConfigHandle CONCAT_EXPAND(FIELD, create_default_fri_transcript_config)() {
    return new FriTranscriptConfig<scalar_t>();
  }

  // struct TranscriptConfigFFI {
  //   HasherHandle merkle_tree_leaves_hash;
  //   std::byte* domain_separator_label;
  //   size_t domain_separator_label_len;
  //   std::byte* round_challenge_label;
  //   size_t round_challenge_label_len;
  //   std::byte* commit_phase_label;
  //   size_t commit_phase_label_len;
  //   std::byte* public_state;
  //   size_t public_state_len;
  //   const scalar_t* seed_rng;
  // };

  FriTranscriptConfigHandle CONCAT_EXPAND(FIELD, create_fri_transcript_config)(
    HasherHandle merkle_tree_leaves_hash,
    std::byte* domain_separator_label,
    size_t domain_separator_label_len,
    std::byte* round_challenge_label,
    size_t round_challenge_label_len,
    std::byte* commit_phase_label,
    size_t commit_phase_label_len,
    std::byte* nonce_label,
    size_t nonce_label_label,
    std::byte* public_state,
    size_t public_state_len,
    scalar_t& seed_rng
  ) {
    std::vector<std::byte> domain_separator_label_vector(domain_separator_label, domain_separator_label + domain_separator_label_len);
    std::vector<std::byte> round_challenge_label_vector(round_challenge_label, round_challenge_label + round_challenge_label_len);
    std::vector<std::byte> commit_phase_label_vector(commit_phase_label, commit_phase_label + commit_phase_label_len);
    std::vector<std::byte> nonce_label_vector(nonce_label, nonce_label + nonce_label_label);
    std::vector<std::byte> public_state_vector(public_state, public_state + public_state_len);
    return new icicle::FriTranscriptConfig<scalar_t>(
      *merkle_tree_leaves_hash, 
      std::move(domain_separator_label_vector), 
      std::move(round_challenge_label_vector), 
      std::move(commit_phase_label_vector), 
      std::move(nonce_label_vector),
      std::move(public_state_vector),
      seed_rng
    );
  }

  eIcicleError icicle_delete_fri_transcript_config(FriTranscriptConfigHandle config_ptr)
  {
    if (!config_ptr) return eIcicleError::INVALID_POINTER;
    delete config_ptr;
    return eIcicleError::SUCCESS;
  }

#ifdef EXT_FIELD

  typedef FriTranscriptConfig<extension_t>* FriTranscriptConfigExtensionHandle;
  typedef FriProof<extension_t>* FriProofExtensionHandle;

  FriProofExtensionHandle CONCAT_EXPAND(FIELD, extension_icicle_initialize_fri_proof)() {
    return new FriProof<extension_t>();
  }

  FriTranscriptConfigExtensionHandle CONCAT_EXPAND(FIELD, extension_create_default_fri_transcript_config)() {
    return new FriTranscriptConfig<extension_t>();
  }
  FriTranscriptConfigExtensionHandle CONCAT_EXPAND(FIELD, extension_create_fri_transcript_config)(
    HasherHandle merkle_tree_leaves_hash,
    std::byte* domain_separator_label,
    size_t domain_separator_label_len,
    std::byte* round_challenge_label,
    size_t round_challenge_label_len,
    std::byte* commit_phase_label,
    size_t commit_phase_label_len,
    std::byte* nonce_label,
    size_t nonce_label_label,
    std::byte* public_state,
    size_t public_state_len,
    extension_t& seed_rng
  ) {
    std::vector<std::byte> domain_separator_label_vector(domain_separator_label, domain_separator_label + domain_separator_label_len);
    std::vector<std::byte> round_challenge_label_vector(round_challenge_label, round_challenge_label + round_challenge_label_len);
    std::vector<std::byte> commit_phase_label_vector(commit_phase_label, commit_phase_label + commit_phase_label_len);
    std::vector<std::byte> nonce_label_vector(nonce_label, nonce_label + nonce_label_label);
    std::vector<std::byte> public_state_vector(public_state, public_state + public_state_len);
    return new icicle::FriTranscriptConfig<extension_t>(
      *merkle_tree_leaves_hash, 
      std::move(domain_separator_label_vector), 
      std::move(round_challenge_label_vector), 
      std::move(commit_phase_label_vector), 
      std::move(nonce_label_vector),
      std::move(public_state_vector),
      seed_rng
    );
  }
#endif

  // FRI

  //           fn icicle_delete_fri_proof(handle: FriProofHandle) -> eIcicleError;

  // FriHandle CONCAT_EXPAND(FIELD, create_fri)(
  //   const size_t input_size,
  //   const size_t folding_factor,
  //   const size_t stopping_degree,
  //   HasherHandle merkle_tree_leaves_hash,
  //   HasherHandle merkle_tree_compress_hash,
  //   const uint64_t output_store_min_layer)
  // {
  //   return new icicle::Fri<scalar_t, scalar_t>(
  //     input_size,
  //     folding_factor,
  //     stopping_degree,
  //     *merkle_tree_leaves_hash,
  //     *merkle_tree_compress_hash,
  //     output_store_min_layer
  //   );
  // }

  // eIcicleError icicle_fri_delete(FriHandle fri_ptr)
  // {
  //   if (!fri_ptr) return eIcicleError::INVALID_POINTER;
  //   delete fri_ptr;
  //   return eIcicleError::SUCCESS;
  // }

  // eIcicleError icicle_fri_get_proof(
  //   const FriHandle fri_ptr,
  //   const FriConfig& fri_config,
  //   const FriTranscriptConfig<scalar_t>& fri_transcript_config,
  //   const scalar_t* input_data,
  //   FriProof<scalar_t>& fri_proof /* OUT */)
  // {
  //   if (!fri_ptr) return eIcicleError::INVALID_POINTER;
  //   return fri_ptr->get_proof(fri_config, fri_transcript_config, input_data, fri_proof);
  // }

  // typedef icicle::FriTranscriptConfig<scalar_t>* FriTranscriptConfigHandle;
  // FriTranscriptConfigHandle CONCAT_EXPAND(FIELD, create_default_fri_transcript_config)() {
  //   return new icicle::FriTranscriptConfig<scalar_t>();
  // }

  // FriTranscriptConfigHandle CONCAT_EXPAND(FIELD, create_fri_transcript_config)(
  //   HasherHandle merkle_tree_leaves_hash, scalar_t& seed_rng
  // ) {
  //   const char* domain_separator_label = "domain_separator_label";
  //   const char* round_challenge_label = "round_challenge_label";
  //   const char* commit_phase_label = "commit_phase_label";
  //   const char* nonce_label = "nonce_label";
  //   std::vector<std::byte>&& public_state = {};
  //   return new icicle::FriTranscriptConfig<scalar_t>(
  //     *merkle_tree_leaves_hash, domain_separator_label, round_challenge_label, commit_phase_label, nonce_label, seed_rng
  //   );
  // }

  // eIcicleError icicle_delete_fri_transcript_config(FriTranscriptConfigHandle config_ptr)
  // {
  //   if (!config_ptr) return eIcicleError::INVALID_POINTER;
  //   delete config_ptr;
  //   return eIcicleError::SUCCESS;
  // }
}