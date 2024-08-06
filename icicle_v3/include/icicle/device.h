#pragma once

#include <cstring>

namespace icicle {

#define MAX_TYPE_LEN 64

  /**
   * @brief Structure representing a device.
   */
  struct Device {
    char type[MAX_TYPE_LEN]; // Type of the device, such as "CPU" or "CUDA"
    int id;                  // Unique identifier for the device (e.g., GPU-2)

    Device(const char* _type, int _id = 0) : id{_id} { copy_str(type, _type); }
    Device(const std::string& _type, int _id = 0) : Device(_type.c_str(), _id) {}
    void copy(const Device& other)
    {
      copy_str(type, other.type);
      id = other.id;
    }
    void copy_str(char* dst, const char* src)
    {
      std::strncpy(dst, src, sizeof(type) - 1);
      dst[sizeof(type) - 1] = 0;
    }
    const Device& operator=(const Device& other)
    {
      copy(other);
      return *this;
    }
    Device(const Device& other) { copy(other); }
    Device(const Device&&) = delete;
    bool operator==(const Device& other) const
    {
      return (id == other.id) && (0 == strncmp(type, other.type, MAX_TYPE_LEN));
    }
    bool operator!=(const Device& other) const { return !(*this == other); }

    friend std::ostream& operator<<(std::ostream& os, const Device& device)
    {
      os << "Device(type: " << device.type << ", id: " << device.id << ")";
      return os;
    }
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
