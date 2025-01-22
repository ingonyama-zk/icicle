#pragma once

#include "icicle/runtime.h"
#include "icicle/errors.h"

namespace icicle {
  /**
   * @brief Configuration structure for pow solve operations.
   *
   * This structure holds configuration options that control how hash operations are executed.
   * It allows specifying the execution stream, challenge location (device or host),
   * and backend-specific extensions. Additionally, it supports both synchronous and asynchronous execution modes.
   */
  struct PowConfig{
    icicleStreamHandle stream = nullptr; /**< Stream for asynchronous execution. Default is nullptr. */
    bool is_challenge_on_device = false; /**< True if challenge reside on the device (e.g., GPU), false if on the host (CPU). Default is false. */
    bool is_result_on_device = false; /**< True if challenge reside on the device (e.g., GPU), false if on the host (CPU). Default is false. */
    bool is_async = false; /**< True to run the pow solver asynchronously, false to run synchronously. Default is false. */
    ConfigExtension* ext = nullptr; /**< Pointer to backend-specific configuration extensions. Default is nullptr. */
  };
  const uint8_t BLOCK_LEN = 64;
  const uint8_t CHALLENGE_LEN = 32;
  const uint8_t PADDING_LEN = 24;

  /**
   * @brief Returns the default value of PowConfig for the POW function.
   *
   * @return Default value of PowConfig.
   */
  static PowConfig default_pow_config() { return PowConfig(); }
  
  eIcicleError pow_blake3(uint8_t* challenge, uint8_t bits, const PowConfig& config, bool& found, uint64_t& nonce, uint64_t& mined_hash);
}