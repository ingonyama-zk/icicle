#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"
#include "icicle/sumcheck/sumcheck.h"

using namespace field_config;

// TODO: Add methods for `prove`, `verify`, and the `proof` struct.

namespace icicle {
  extern "C" {

  // Define the Sumcheck handle type
  typedef Sumcheck<scalar_t> SumcheckHandle;

  // Structure to represent the FFI transcript configuration
  struct TranscriptConfigFFI {
    Hash* hasher;
    std::byte* domain_separator_label;
    size_t domain_separator_label_len;
    std::byte* round_poly_label;
    size_t round_poly_label_len;
    std::byte* round_challenge_label;
    size_t round_challenge_label_len;
    bool little_endian;
    const scalar_t* seed_rng;
  };

  /**
   * @brief Creates a new Sumcheck instance from the given FFI transcript configuration.
   * @param ffi_transcript_config Pointer to the FFI transcript configuration structure.
   * @return Pointer to the created Sumcheck instance.
   */
  SumcheckHandle* CONCAT_EXPAND(FIELD, sumcheck_create)(const TranscriptConfigFFI* ffi_transcript_config)
  {
    if (!ffi_transcript_config || !ffi_transcript_config->hasher || !ffi_transcript_config->seed_rng) {
      ICICLE_LOG_ERROR << "Invalid FFI transcript configuration.";
      return nullptr;
    }

    ICICLE_LOG_DEBUG << "Constructing Sumcheck instance from FFI";

    // Convert byte arrays to vectors
    std::vector<std::byte> domain_separator_label(
      ffi_transcript_config->domain_separator_label,
      ffi_transcript_config->domain_separator_label + ffi_transcript_config->domain_separator_label_len);
    std::vector<std::byte> round_poly_label(
      ffi_transcript_config->round_poly_label,
      ffi_transcript_config->round_poly_label + ffi_transcript_config->round_poly_label_len);
    std::vector<std::byte> round_challenge_label(
      ffi_transcript_config->round_challenge_label,
      ffi_transcript_config->round_challenge_label + ffi_transcript_config->round_challenge_label_len);

    // Construct the SumcheckTranscriptConfig
    SumcheckTranscriptConfig config{*ffi_transcript_config->hasher,   std::move(domain_separator_label),
                                    std::move(round_poly_label),      std::move(round_challenge_label),
                                    *ffi_transcript_config->seed_rng, ffi_transcript_config->little_endian};

    // Create and return the Sumcheck instance
    return new Sumcheck<scalar_t>(std::move(config));
  }

  /**
   * @brief Deletes the given Sumcheck instance.
   * @param sumcheck_handle Pointer to the Sumcheck instance to be deleted.
   * @return eIcicleError indicating the success or failure of the operation.
   */
  eIcicleError CONCAT_EXPAND(FIELD, sumcheck_delete)(const SumcheckHandle* sumcheck_handle)
  {
    if (!sumcheck_handle) {
      ICICLE_LOG_ERROR << "Cannot delete a null Sumcheck instance.";
      return eIcicleError::INVALID_ARGUMENT;
    }

    ICICLE_LOG_DEBUG << "Destructing Sumcheck instance from FFI";
    delete sumcheck_handle;

    return eIcicleError::SUCCESS;
  }

  } // extern "C"

} // namespace icicle