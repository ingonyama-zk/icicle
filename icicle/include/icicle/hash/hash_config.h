#pragma once

#include "icicle/runtime.h"
#include "icicle/config_extension.h"

namespace icicle {

  /**
   * @brief Configuration structure for hash operations.
   *
   * This structure holds various configuration options that control how hash operations are executed.
   * It allows specifying the execution stream, input/output locations (device or host), batch sizes,
   * and backend-specific extensions. Additionally, it supports both synchronous and asynchronous execution modes.
   */
  struct HashConfig {
    icicleStreamHandle stream = nullptr; /**< Stream for asynchronous execution. Default is nullptr. */
    uint64_t batch = 1;                  /**< Number of hashes to perform in parallel (independently). Default is 1. */
    bool are_inputs_on_device =
      false; /**< True if inputs reside on the device (e.g., GPU), false if on the host (CPU). Default is false. */
    bool are_outputs_on_device =
      false;               /**< True if outputs reside on the device, false if on the host. Default is false. */
    bool is_async = false; /**< True to run the hash asynchronously, false to run synchronously. Default is false. */
    ConfigExtension* ext = nullptr; /**< Pointer to backend-specific configuration extensions. Default is nullptr. */
  };

  /**
   * @brief Creates a default HashConfig object.
   *
   * This function provides a default configuration for hash operations. The default configuration
   * runs the hash synchronously on the host, with a single chunk of input and no backend-specific extensions.
   * @return A default-initialized HashConfig.
   */
  static HashConfig default_hash_config() { return HashConfig(); }

} // namespace icicle