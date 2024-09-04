#pragma once

#include "icicle/runtime.h"
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
    icicleStreamHandle stream;  /**< Stream for asynchronous execution. */
    bool are_inputs_on_device;  ///< True if inputs are on the device, false if on the host. Default is false.
    bool are_outputs_on_device; ///< True if outputs are on the device, false if on the host. Default is false.
    bool is_async; ///< True to run the hash asynchronously, false to run it synchronously. Default is false.
    ConfigExtension* ext =
      nullptr; ///< Backend-specific extensions. This allows customization for specific device backends.
  };

  static HashConfig default_hash_config()
  {
    HashConfig config = {
      nullptr, // stream
      false,   // are_inputs_on_device
      false,   // are_outputs_on_device
      false,   // is_async
      nullptr  // ext
    };
    return config;
  }

} // namespace icicle