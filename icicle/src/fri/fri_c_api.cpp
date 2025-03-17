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

  FriProofHandle CONCAT_EXPAND(ICICLE_FFI_PREFIX, icicle_initialize_fri_proof)() {
    return new FriProof<scalar_t>();
  }

  eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, icicle_delete_fri_proof)(FriProofHandle proof_ptr) {
    if (!proof_ptr) return eIcicleError::INVALID_POINTER;
    delete proof_ptr;
    return eIcicleError::SUCCESS;
  }

  eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_get_proof_sizes)(FriProofHandle proof_ptr, size_t& nof_queries, size_t& nof_rounds) {
    if (!proof_ptr) {
      ICICLE_LOG_ERROR << "CANNOT USE null proof_ptr";
      return eIcicleError::INVALID_POINTER;
    }
    nof_queries = proof_ptr->get_nof_fri_queries();
    nof_rounds = proof_ptr->get_nof_fri_rounds();
    return eIcicleError::SUCCESS;
  }

  eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_get_round_proof_at)(FriProofHandle proof_ptr, size_t query_idx, MerkleProof** proofs) {
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


  eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_get_final_poly_size)(FriProofHandle proof_ptr, size_t& result) {
    if (!proof_ptr) {
      ICICLE_LOG_ERROR << "CANNOT USE null proof_ptr";
      return eIcicleError::INVALID_POINTER;
    }
    result = proof_ptr->get_final_poly_size();
    return eIcicleError::SUCCESS;
  }

  eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_get_final_poly)(FriProofHandle proof_ptr, scalar_t** final_poly) {
    if (!proof_ptr) {
      ICICLE_LOG_ERROR << "CANNOT USE null proof_ptr";
      return eIcicleError::INVALID_POINTER;
    }
    *final_poly = proof_ptr->get_final_poly();
    return eIcicleError::SUCCESS;
  }

  eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, fri_proof_get_pow_nonce)(FriProofHandle proof_ptr, uint64_t& result) {
    if (!proof_ptr) {
      ICICLE_LOG_ERROR << "CANNOT USE null proof_ptr";
      return eIcicleError::INVALID_POINTER;
    }
    result = proof_ptr->get_pow_nonce();
    return eIcicleError::SUCCESS;
  }

#ifdef EXT_FIELD

  typedef FriTranscriptConfig<extension_t>* FriTranscriptConfigExtensionHandle;
  typedef FriProof<extension_t>* FriProofExtensionHandle;

  FriProofExtensionHandle CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_icicle_initialize_fri_proof)() {
    return new FriProof<extension_t>();
  }

  eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_icicle_delete_fri_proof)(FriProofExtensionHandle proof_ptr) {
      if (!proof_ptr) return eIcicleError::INVALID_POINTER;
      delete proof_ptr;
      return eIcicleError::SUCCESS;
  }

  eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_get_proof_sizes)(FriProofExtensionHandle proof_ptr, size_t& nof_queries, size_t& nof_rounds) {
      if (!proof_ptr) {
          ICICLE_LOG_ERROR << "CANNOT USE null proof_ptr";
          return eIcicleError::INVALID_POINTER;
      }
      nof_queries = proof_ptr->get_nof_fri_queries();
      nof_rounds = proof_ptr->get_nof_fri_rounds();
      return eIcicleError::SUCCESS;
  }

  eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_get_round_proof_at)(FriProofExtensionHandle proof_ptr, size_t query_idx, MerkleProof** proofs) {
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

  eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_get_final_poly_size)(FriProofExtensionHandle proof_ptr, size_t& result) {
      if (!proof_ptr) {
          ICICLE_LOG_ERROR << "CANNOT USE null proof_ptr";
          return eIcicleError::INVALID_POINTER;
      }
      result = proof_ptr->get_final_poly_size();
      return eIcicleError::SUCCESS;
  }

  eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_get_final_poly)(FriProofExtensionHandle proof_ptr, extension_t** final_poly) {
      if (!proof_ptr) {
          ICICLE_LOG_ERROR << "CANNOT USE null proof_ptr";
          return eIcicleError::INVALID_POINTER;
      }
      *final_poly = proof_ptr->get_final_poly();
      return eIcicleError::SUCCESS;
  }

  eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, extension_fri_proof_get_pow_nonce)(FriProofExtensionHandle proof_ptr, uint64_t& result) {
      if (!proof_ptr) {
          ICICLE_LOG_ERROR << "CANNOT USE null proof_ptr";
          return eIcicleError::INVALID_POINTER;
      }
      result = proof_ptr->get_pow_nonce();
      return eIcicleError::SUCCESS;
  }
#endif

}