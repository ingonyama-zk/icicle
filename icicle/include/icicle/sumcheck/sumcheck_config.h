#pragma once

#include "icicle/runtime.h"
#include "icicle/config_extension.h"

namespace icicle {
  /**
   * @brief Configuration structure for Sumcheck operations.
   *
   * This structure holds the configuration options for sumcheck operations.
   * It allows specifying whether the data (input MLE polynomials)
   * reside on the device (e.g., GPU) or the host (e.g., CPU), and supports both synchronous and asynchronous
   * execution modes, as well as backend-specific extensions.
   */

  struct SumcheckConfig {
    icicleStreamHandle stream = nullptr; /**< Stream for asynchronous execution. Default is nullptr. */
    bool use_extension_field = false; /**< If true, then use extension field for the fiat shamir result. Recommended for
                                         small fields for security*/
    uint64_t batch = 1;               /**< Number of input chunks to hash in batch. Default is 1. */
    bool are_inputs_on_device =
      false; /**< True if inputs reside on the device (e.g., GPU), false if on the host (CPU). Default is false. */
    bool is_async = false; /**< True to run the hash asynchronously, false to run synchronously. Default is false. */
    ConfigExtension* ext = nullptr; /**< Pointer to backend-specific configuration extensions. Default is nullptr. */
  };

  /**
   * @brief Generates a default configuration for Sumcheck operations.
   *
   * This function provides a default configuration for Sumcheck operations with synchronous execution
   * and all data (leaves, tree results, and paths) residing on the host (CPU).
   *
   * @return A default SumcheckConfig with host-based execution and no backend-specific extensions.
   */
  static SumcheckConfig default_sumcheck_config() { return SumcheckConfig(); }

} // namespace icicle
