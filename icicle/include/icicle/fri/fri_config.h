#pragma once

#include "icicle/runtime.h"
#include "icicle/config_extension.h"

namespace icicle {

  /**
   * @brief Configuration structure for FRI operations.
   *
   * This structure holds the configuration options for FRI operations.
   * It provides control over proof-of-work requirements, query iterations,
   * execution modes (synchronous/asynchronous), and device/host data placement.
   * It also supports backend-specific extensions for customization.
   */
  struct FriConfig {
    icicleStreamHandle stream = nullptr; // Stream for asynchronous execution.
    size_t folding_factor = 2;           // The factor by which the codeword is folded in each round.
    size_t stopping_degree = 0;          // The minimal polynomial degree at which folding stops.
    size_t pow_bits = 16;                // Number of leading zeros required for proof-of-work.
    size_t nof_queries = 100;            // Number of queries, computed for each folded layer of FRI.
    bool are_inputs_on_device =
      false; // True if inputs reside on the device (e.g., GPU), false if on the host (CPU).
    bool is_async = false; // True to run operations asynchronously, false to run synchronously.
    ConfigExtension* ext = nullptr; // Pointer to backend-specific configuration extensions.
  };

  /**
   * @brief Generates a default configuration for FRI operations.
   *
   * This function provides a default configuration for FRI operations with synchronous execution
   * and all data (inputs, outputs, etc.) residing on the host (CPU).
   *
   * @return A default FriConfig with host-based execution and no backend-specific extensions.
   */
  static FriConfig default_fri_config() { return FriConfig(); }

} // namespace icicle
