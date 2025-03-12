#include <memory>
#include "icicle/fri/fri.h"
#include "icicle/fri/fri_transcript_config.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"

using namespace field_config;
using namespace icicle;

extern "C" {
  typedef icicle::Fri<scalar_t, scalar_t>* FriHandle;
  typedef icicle::Hash* HasherHandle;

  FriHandle CONCAT_EXPAND(FIELD, create_fri)(
    const size_t input_size,
    const size_t folding_factor,
    const size_t stopping_degree,
    HasherHandle merkle_tree_leaves_hash,
    HasherHandle merkle_tree_compress_hash,
    const uint64_t output_store_min_layer)
  {
    return new icicle::Fri<scalar_t, scalar_t>(
      input_size,
      folding_factor,
      stopping_degree,
      *merkle_tree_leaves_hash,
      *merkle_tree_compress_hash,
      output_store_min_layer
    );
  }

  eIcicleError icicle_fri_delete(FriHandle fri_ptr)
  {
    if (!fri_ptr) return eIcicleError::INVALID_POINTER;
    delete fri_ptr;
    return eIcicleError::SUCCESS;
  }

  eIcicleError icicle_fri_get_proof(
    const FriHandle fri_ptr,
    const FriConfig& fri_config,
    const FriTranscriptConfig<scalar_t>& fri_transcript_config,
    const scalar_t* input_data,
    FriProof<scalar_t>& fri_proof /* OUT */)
  {
    if (!fri_ptr) return eIcicleError::INVALID_POINTER;
    return fri_ptr->get_proof(fri_config, fri_transcript_config, input_data, fri_proof);
  }

  typedef icicle::FriTranscriptConfig<scalar_t>* FriTranscriptConfigHandle;
  FriTranscriptConfigHandle CONCAT_EXPAND(FIELD, create_default_fri_transcript_config)() {
    return new icicle::FriTranscriptConfig<scalar_t>();
  }

  FriTranscriptConfigHandle CONCAT_EXPAND(FIELD, create_fri_transcript_config)(
    HasherHandle merkle_tree_leaves_hash, scalar_t& seed_rng
  ) {
    const char* domain_separator_label = "domain_separator_label";
    const char* round_challenge_label = "round_challenge_label";
    const char* commit_phase_label = "commit_phase_label";
    const char* nonce_label = "nonce_label";
    std::vector<std::byte>&& public_state = {};
    return new icicle::FriTranscriptConfig<scalar_t>(
      *merkle_tree_leaves_hash, domain_separator_label, round_challenge_label, commit_phase_label, nonce_label, seed_rng
    );
  }

  eIcicleError icicle_delete_fri_transcript_config(FriTranscriptConfigHandle config_ptr)
  {
    if (!config_ptr) return eIcicleError::INVALID_POINTER;
    delete config_ptr;
    return eIcicleError::SUCCESS;
  }
}