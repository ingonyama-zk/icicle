#pragma once

#include "icicle/runtime.h"
#include "icicle/errors.h"
#include "icicle/hash/hash.h"
#include "icicle/config_extension.h"

namespace icicle {
  /**
   * @brief Configuration structure for pow solve operations.
   *
   * This structure holds configuration options that control how hash operations are executed.
   * It allows specifying the execution stream, challenge location (device or host),
   * and backend-specific extensions. Additionally, it supports both synchronous and asynchronous execution modes.
   */
  struct PowConfig {
    icicleStreamHandle stream = nullptr; /**< Stream for asynchronous execution. Default is nullptr. */
    bool is_challenge_on_device =
      false; /**< True if challenge reside on the device (e.g., GPU), false if on the host (CPU). Default is false. */
    uint32_t padding_size =
      24; /**< Size of padding (in bytes) applied to the challenge before hashing. Default is 24. */
    bool is_async =
      false; /**< True to run the pow solver asynchronously, false to run synchronously. Default is false. */
    ConfigExtension* ext = nullptr; /**< Pointer to backend-specific configuration extensions. Default is nullptr. */
  };

  /**
   * @brief Returns the default value of PowConfig for the POW function.
   *
   * @return Default value of PowConfig.
   */
  static PowConfig default_pow_config() { return PowConfig(); }

  extern "C" {
  /**
   * @brief Solves the proof-of-work (PoW) challenge using the given hashing algorithm.
   *
   * This function attempts to find a valid nonce that satisfies the proof-of-work
   * requirements by iterating over possible values and hashing them.
   *
   * @param hasher The hashing algorithm to be used.
   * @param challenge A pointer to the challenge data.
   * @param challenge_size The size of the challenge data in bytes.
   * @param solution_bits The number of leading zero bits required in the hash solution.
   * @param config The configuration parameters for the PoW process.
   * @param found A reference to a boolean flag that is set to true if a valid solution is found.
   * @param nonce A reference to the nonce value that produces a valid hash (if found).
   * @param mined_hash A reference to the resulting hash value of the successful PoW attempt.
   *
   * @return eIcicleError Error code indicating success or failure of the operation.
   */
  eIcicleError proof_of_work(
    const Hash& hasher,
    const std::byte* challenge,
    uint32_t challenge_size,
    uint8_t solution_bits,
    const PowConfig& config,
    bool& found,
    uint64_t& nonce,
    uint64_t& mined_hash);

  /**
   * @brief Verifies the proof-of-work (PoW) solution for a given challenge.
   *
   * This function checks whether the provided nonce produces a valid hash that meets
   * the required difficulty (number of leading zero bits).
   *
   * @param hasher The hashing algorithm to be used.
   * @param challenge A pointer to the challenge data.
   * @param challenge_size The size of the challenge data in bytes.
   * @param solution_bits The number of leading zero bits required in the hash solution.
   * @param config The configuration parameters for the PoW process.
   * @param nonce The nonce value to be verified.
   * @param is_correct A reference to a boolean flag set to true if the nonce produces a valid hash.
   * @param mined_hash A reference to store the computed hash value for verification.
   *
   * @return eIcicleError Error code indicating success or failure of the verification process.
   */
  eIcicleError proof_of_work_verify(
    const Hash& hasher,
    const std::byte* challenge,
    uint32_t challenge_size,
    uint8_t solution_bits,
    const PowConfig& config,
    uint64_t nonce,
    bool& is_correct,
    uint64_t& mined_hash);
  }
} // namespace icicle