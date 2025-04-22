#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"
#include "icicle/sumcheck/sumcheck.h"
#include "icicle/serialization.h"

using namespace field_config;

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

/**************** BEGIN Sumcheck ***************************/
/**
 * @brief Creates a new Sumcheck instance from the given FFI transcript configuration.
 * @param ffi_transcript_config Pointer to the FFI transcript configuration structure.
 * @return Pointer to the created Sumcheck instance.
 */
SumcheckHandle* CONCAT_EXPAND(ICICLE_FFI_PREFIX, sumcheck_create)()
{
  ICICLE_LOG_DEBUG << "Constructing Sumcheck instance from FFI";
  // Create and return the Sumcheck instance
  return new icicle::Sumcheck<scalar_t>(icicle::create_sumcheck<scalar_t>());
}

/**
 * @brief Deletes the given Sumcheck instance.
 * @param sumcheck_handle Pointer to the Sumcheck instance to be deleted.
 * @return eIcicleError indicating the success or failure of the operation.
 */
eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, sumcheck_delete)(const SumcheckHandle* sumcheck_handle)
{
  if (!sumcheck_handle) {
    ICICLE_LOG_ERROR << "Cannot delete a null Sumcheck instance.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  ICICLE_LOG_DEBUG << "Destructing Sumcheck instance from FFI";
  delete sumcheck_handle;

  return eIcicleError::SUCCESS;
}

/**
 * @brief Creates the sumcheck proof of the claimed sum
 * @param sumcheck_handle - The pointer to the Sumcheck instance
 * @param ffi_mle_polynomials - The polynomials used for the proof
 * @param mle_polynomial_size - The size of the polynomials
 * @param nof_mle_polynomials - The number of polynomials
 * @param claimed_sum - The claimed sum that is trying to be proven
 * @param combine_function - The combine function to use in the Sumcheck protocol
 * @param ffi_transcript_config - The transcript config to use in the Sumcheck protocol
 * @param sumcheck_config - The sumcheck config to use in the Sumcheck protocol
 * @return SumcheckProof<scalar_t>* a pointer to the SumcheckProof that contains the round_polynomials.
 */
SumcheckProof<scalar_t>* CONCAT_EXPAND(ICICLE_FFI_PREFIX, sumcheck_get_proof)(
  const SumcheckHandle* sumcheck_handle,
  const scalar_t** ffi_mle_polynomials,
  const uint64_t mle_polynomial_size,
  const uint64_t nof_mle_polynomials,
  const scalar_t* claimed_sum,
  const CombineFunction<scalar_t>* combine_function,
  const TranscriptConfigFFI* ffi_transcript_config,
  const SumcheckConfig* sumcheck_config)
{
  // Start constructing the MLE polynomials
  std::vector<scalar_t*> mle_polynomials(nof_mle_polynomials);

  for (auto i = 0; i < nof_mle_polynomials; i++) {
    mle_polynomials[i] = const_cast<scalar_t*>(*(ffi_mle_polynomials + i));
  }
  // Finished constructing the MLE polynomials

  // Start constructing the SumcheckTranscriptConfig from TranscriptConfigFFI
  if (!ffi_transcript_config || !ffi_transcript_config->hasher || !ffi_transcript_config->seed_rng) {
    ICICLE_LOG_ERROR << "Invalid FFI transcript configuration.";
    return nullptr;
  }

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

  SumcheckTranscriptConfig transcript_config{*ffi_transcript_config->hasher,   std::move(domain_separator_label),
                                             std::move(round_poly_label),      std::move(round_challenge_label),
                                             *ffi_transcript_config->seed_rng, ffi_transcript_config->little_endian};
  // Finished constructing the SumcheckTranscriptConfig from TranscriptConfigFFI

  SumcheckProof<scalar_t>* sumcheck_proof = new SumcheckProof<scalar_t>();
  sumcheck_handle->get_proof(
    mle_polynomials, mle_polynomial_size, *claimed_sum, *combine_function, std::move(transcript_config),
    *sumcheck_config, *sumcheck_proof);

  return sumcheck_proof;
}

/**
 * @brief Verify a given sumcheck proof
 * @param sumcheck_handle - The pointer to the Sumcheck instance.
 * @param sumcheck_proof_handle - The pointer to the SumcheckProof that will be proven.
 * @param claimed_sum - The claimed sum that is trying to be proven
 * @param ffi_transcript_config - The transcript config to use in the Sumcheck protocol
 * @return eIcicleError indicating the success or failure of the operation.
 */
eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, sumcheck_verify)(
  SumcheckHandle* sumcheck_handle,
  const SumcheckProof<scalar_t>* sumcheck_proof_handle,
  const scalar_t* claimed_sum,
  const TranscriptConfigFFI* ffi_transcript_config,
  bool* is_verified)
{
  if (!ffi_transcript_config || !ffi_transcript_config->hasher || !ffi_transcript_config->seed_rng) {
    ICICLE_LOG_ERROR << "Invalid FFI transcript configuration.";
    return eIcicleError::INVALID_POINTER;
  }

  ICICLE_LOG_DEBUG << "Verifying SumcheckProof from FFI";

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
  SumcheckTranscriptConfig transcript_config{*ffi_transcript_config->hasher,   std::move(domain_separator_label),
                                             std::move(round_poly_label),      std::move(round_challenge_label),
                                             *ffi_transcript_config->seed_rng, ffi_transcript_config->little_endian};

  return sumcheck_handle->verify(
    *sumcheck_proof_handle, *claimed_sum, std::move(transcript_config), *is_verified /*out*/
  );
}
/**************** END Sumcheck ***************************/

/**************** BEGIN SumcheckProof ***************************/
/**
 * @brief Creates a new SumcheckProof instance from the given FFI transcript configuration.
 * @param polys - The round polynomials of the proof
 * @param nof_polynomials - The number of round polynomials in the proof
 * @param poly_size - The size of the round polynomials of the proof
 * @return Pointer to the created SumcheckProof instance.
 */
SumcheckProof<scalar_t>* CONCAT_EXPAND(ICICLE_FFI_PREFIX, sumcheck_proof_create)(
  scalar_t** polys, const uint64_t nof_polynomials, const uint64_t poly_size)
{
  ICICLE_LOG_DEBUG << "Reconstructing SumcheckProof from values from FFI";
  // Start constructing SumcheckProof from the round_polynomials
  std::vector<std::vector<scalar_t>> round_polynomials(nof_polynomials);

  for (auto i = 0; i < nof_polynomials; i++) {
    round_polynomials[i] = {*(polys + i), *(polys + i) + poly_size};
  }

  return new icicle::SumcheckProof<scalar_t>(round_polynomials);
}

/**
 * @brief Obtains the proof's round polynomial metadata
 * @param sumcheck_proof_handle Pointer to the SumcheckProof instance.
 * @param poly_size - Pointer to store the size of the round polynomials of the proof
 * @param nof_polys - Pointer to store the number of round polynomials in the proof
 * @return eIcicleError indicating the success or failure of the operation.
 */
eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, sumcheck_proof_get_poly_sizes)(
  SumcheckProof<scalar_t>* sumcheck_proof_handle, uint64_t* poly_size, uint64_t* nof_polys)
{
  if (!sumcheck_proof_handle) {
    ICICLE_LOG_ERROR << "SumcheckProofHandle is null.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  *poly_size = sumcheck_proof_handle->get_round_polynomial_size() - 1;
  *nof_polys = sumcheck_proof_handle->get_nof_round_polynomials();

  return eIcicleError::SUCCESS;
}

/**
 * @brief Obtains the round_polynomial at a given index.
 * @param sumcheck_proof_handle Pointer to the SumcheckProof instance.
 * @param index The round_polynomial index to get.
 * @return eIcicleError indicating the success or failure of the operation.
 */
scalar_t* CONCAT_EXPAND(ICICLE_FFI_PREFIX, sumcheck_proof_get_round_poly_at)(
  SumcheckProof<scalar_t>* sumcheck_proof_handle, uint64_t index)
{
  if (!sumcheck_proof_handle) {
    ICICLE_LOG_ERROR << "Cannot delete a null SumcheckProof instance.";
    return nullptr;
  }

  uint nof_polys = sumcheck_proof_handle->get_nof_round_polynomials();

  if (index >= nof_polys) {
    ICICLE_LOG_ERROR << "Index is greater than number of round_polys in proof";
    return nullptr;
  }

  return sumcheck_proof_handle->get_round_polynomial(index).data();
}

/**
 * @brief Deletes the given SumcheckProof instance.
 * @param sumcheck_proof_handle Pointer to the SumcheckProof instance to be deleted.
 * @return eIcicleError indicating the success or failure of the operation.
 */
eIcicleError
CONCAT_EXPAND(ICICLE_FFI_PREFIX, sumcheck_proof_delete)(const SumcheckProof<scalar_t>* sumcheck_proof_handle)
{
  if (!sumcheck_proof_handle) {
    ICICLE_LOG_ERROR << "Cannot delete a null SumcheckProof instance.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  ICICLE_LOG_DEBUG << "Destructing SumcheckProof instance from FFI";
  delete sumcheck_proof_handle;

  return eIcicleError::SUCCESS;
}

/**
 * @brief Prints the given SumcheckProof instance.
 * @param sumcheck_proof_handle Pointer to the SumcheckProof instance to print.
 * @return eIcicleError indicating the success or failure of the operation.
 */
eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, sumcheck_proof_print)(SumcheckProof<scalar_t>* sumcheck_proof_handle)
{
  if (!sumcheck_proof_handle) {
    ICICLE_LOG_ERROR << "Cannot delete a null SumcheckProof instance.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  ICICLE_LOG_DEBUG << "Destructing SumcheckProof instance from FFI";
  sumcheck_proof_handle->print_proof();

  return eIcicleError::SUCCESS;
}

/**
 * @brief Gets the serialized size of the given SumcheckProof instance.
 * @param sumcheck_proof_handle Pointer to the SumcheckProof instance.
 * @param size Pointer to store the serialized size of the SumcheckProof instance.
 * @return eIcicleError indicating the success or failure of the operation.
 */
eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, sumcheck_proof_get_serialized_size)(
  SumcheckProof<scalar_t>* sumcheck_proof_handle, size_t* size)
{
  if (!sumcheck_proof_handle) {
    ICICLE_LOG_ERROR << "Cannot get serialized size of a null SumcheckProof instance.";
    return eIcicleError::INVALID_ARGUMENT;
  }
  if (!size) {
    ICICLE_LOG_ERROR << "Cannot write serialized size in a null pointer.";
    return eIcicleError::INVALID_ARGUMENT;
  }
  return BinarySerializer<SumcheckProof<scalar_t>>::serialized_size(*sumcheck_proof_handle, *size);
}

/**
 * @brief Serializes the given SumcheckProof instance to a buffer.
 * @param sumcheck_proof_handle Pointer to the SumcheckProof instance to serialize.
 * @param buffer The buffer to serialize the SumcheckProof to.
 * @param size The size of the buffer.
 * @return eIcicleError indicating the success or failure of the operation.
 */
eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, sumcheck_proof_serialize)(
  SumcheckProof<scalar_t>* sumcheck_proof_handle, std::byte* buffer, size_t size)
{
  if (!sumcheck_proof_handle) {
    ICICLE_LOG_ERROR << "Cannot serialize a null SumcheckProof instance.";
    return eIcicleError::INVALID_POINTER;
  }
  if (!buffer) {
    ICICLE_LOG_ERROR << "Cannot serialize to a null buffer.";
    return eIcicleError::INVALID_POINTER;
  }
  size_t expected_size = 0;
  eIcicleError err = BinarySerializer<SumcheckProof<scalar_t>>::serialized_size(*sumcheck_proof_handle, expected_size);
  if (err != eIcicleError::SUCCESS) {
    ICICLE_LOG_ERROR << "Cannot get serialized size of SumcheckProof";
    return err;
  }
  if (size < expected_size) {
    ICICLE_LOG_ERROR << "buffer is too small — cannot serialize SumcheckProof";
    return eIcicleError::INVALID_ARGUMENT;
  }
  return BinarySerializer<SumcheckProof<scalar_t>>::serialize(buffer, size, *sumcheck_proof_handle);
}

/**
 * @brief Deserializes the given SumcheckProof instance from a buffer.
 * @param sumcheck_proof_handle Pointer to the SumcheckProof instance to deserialize.
 * @param buffer The buffer to deserialize the SumcheckProof from.
 * @param size The size of the buffer.
 * @return eIcicleError indicating the success or failure of the operation.
 */
eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, sumcheck_proof_deserialize)(
  SumcheckProof<scalar_t>** sumcheck_proof_handle, std::byte* buffer, size_t size)
{
  if (!sumcheck_proof_handle) {
    ICICLE_LOG_ERROR << "Cannot deserialize into a null SumcheckProof instance.";
    return eIcicleError::INVALID_ARGUMENT;
  }
  if (!buffer) {
    ICICLE_LOG_ERROR << "Cannot deserialize from a null buffer.";
    return eIcicleError::INVALID_POINTER;
  }
  *sumcheck_proof_handle = new SumcheckProof<scalar_t>();
  return BinarySerializer<SumcheckProof<scalar_t>>::deserialize(buffer, size, *sumcheck_proof_handle);
}

/**
 * @brief Serializes the given SumcheckProof instance to a file.
 * @param sumcheck_proof_handle Pointer to the SumcheckProof instance to serialize.
 * @param filename The name of the file to serialize the SumcheckProof to.
 * @return eIcicleError indicating the success or failure of the operation.
 */
eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, sumcheck_proof_serialize_to_file)(
  SumcheckProof<scalar_t>* sumcheck_proof_handle, const char* filename, size_t filename_len)
{
  if (!sumcheck_proof_handle) {
    ICICLE_LOG_ERROR << "Cannot serialize a null SumcheckProof instance.";
    return eIcicleError::INVALID_ARGUMENT;
  }
  if (!filename || !filename_len) {
    ICICLE_LOG_ERROR << "Cannot serialize to a null filename.";
    return eIcicleError::INVALID_ARGUMENT;
  }
  std::string filename_str(filename, filename_len);
  return BinarySerializer<SumcheckProof<scalar_t>>::serialize_to_file(std::move(filename_str), *sumcheck_proof_handle);
}

/**
 * @brief Deserializes the given SumcheckProof instance from a file.
 * @param sumcheck_proof_handle Pointer to the SumcheckProof instance to deserialize.
 * @param filename The name of the file to deserialize the SumcheckProof from.
 * @return eIcicleError indicating the success or failure of the operation.
 */
eIcicleError CONCAT_EXPAND(ICICLE_FFI_PREFIX, sumcheck_proof_deserialize_from_file)(
  SumcheckProof<scalar_t>** sumcheck_proof_handle, const char* filename, size_t filename_len)
{
  if (!sumcheck_proof_handle) {
    ICICLE_LOG_ERROR << "Cannot deserialize into a null SumcheckProof instance.";
    return eIcicleError::INVALID_ARGUMENT;
  }
  if (!filename || !filename_len) {
    ICICLE_LOG_ERROR << "Cannot deserialize from a null filename.";
    return eIcicleError::INVALID_ARGUMENT;
  }
  std::string filename_str(filename, filename_len);
  *sumcheck_proof_handle = new SumcheckProof<scalar_t>();
  return BinarySerializer<SumcheckProof<scalar_t>>::deserialize_from_file(
    std::move(filename_str), *sumcheck_proof_handle);
}

/***************** END SumcheckProof **********************/
} // extern "C"
