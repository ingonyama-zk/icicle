#pragma once

namespace icicle {

  struct Device {
    const char* type; // Type of the device, such as "CPU" or "CUDA"
    int id;           // Unique identifier for the device (e.g., GPU-2)
  };

}; // namespace icicle