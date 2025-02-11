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
    icicleStreamHandle stream = nullptr;      // Stream for asynchronous execution. Default is nullptr.
    bool use_extension_field = false;         // If true, then use extension field for the fiat shamir result. Recommended for small fields for security. TODO SHANIE - this was not part of the design plan, do we need to add this like in sumcheck?
    size_t pow_bits = 0;                      // Number of leading zeros required for proof-of-work. Default is 0.
    size_t nof_queries = 1;                   // Number of queries, computed for each folded layer of FRI. Default is 1.
    bool are_inputs_on_device = false;        // True if inputs reside on the device (e.g., GPU), false if on the host (CPU). Default is false.
    bool are_outputs_on_device = false;       // True if outputs reside on the device, false if on the host. Default is false.
    bool is_async = false;                    // True to run operations asynchronously, false to run synchronously. Default is false.
    ConfigExtension* ext = nullptr;           // Pointer to backend-specific configuration extensions. Default is nullptr.
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
