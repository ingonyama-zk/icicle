#pragma once

namespace icicle {

  /**
   * @brief Structure representing a device.
   */
  struct Device {
    const char* type; // Type of the device, such as "CPU" or "CUDA"
    int id;           // Unique identifier for the device (e.g., GPU-2)
  };

  /**
   * @brief Structure to hold device properties.
   */
  struct DeviceProperties {
    bool using_host_memory;      // Indicates if the device uses host memory
    int num_memory_regions;      // Number of memory regions available on the device
    bool supports_pinned_memory; // Indicates if the device supports pinned memory
    // Add more properties as needed
  };

} // namespace icicle
