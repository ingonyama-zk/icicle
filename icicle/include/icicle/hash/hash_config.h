#pragma once

#include "icicle/config_extension.h"

namespace icicle {

  /**
   * @brief Configuration struct for the hash.
   *
   * This struct holds configuration options for hash operations. It allows the user
   * to specify whether inputs and outputs reside on the device (e.g., GPU) or the host (e.g., CPU),
   * whether the hash operation should run asynchronously, and any backend-specific extensions.
   */
  struct HashConfig {
    bool are_inputs_on_device = false;  ///< True if inputs are on the device, false if on the host. Default is false.
    bool are_outputs_on_device = false; ///< True if outputs are on the device, false if on the host. Default is false.
    bool is_async = false; ///< True to run the hash asynchronously, false to run it synchronously. Default is false.
    ConfigExtension* ext =
      nullptr; ///< Backend-specific extensions. This allows customization for specific device backends.
  };

} // namespace icicle