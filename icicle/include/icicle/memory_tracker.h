#pragma once

#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <thread>

#include "icicle/device.h"

namespace icicle {

  class MemoryTracker
  {
  public:
    // Add a new allocation
    void add_allocation(void* address, size_t size, const Device& device)
    {
      std::lock_guard<std::mutex> lock(mutex_);
      allocations_.insert(std::make_pair(address, AllocationInfo{size, device}));
    }

    // Remove an allocation
    void remove_allocation(void* address)
    {
      std::lock_guard<std::mutex> lock(mutex_);
      allocations_.erase(address);
    }

    // Check if an address is allocated by this tracker and get the device
    std::optional<const Device*> identify_device(const void* address)
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = allocations_.upper_bound(address);
      if (it == allocations_.begin()) { return std::nullopt; }
      --it;
      const char* start = static_cast<const char*>(it->first);
      const char* end = start + it->second.size_;
      if (start <= static_cast<const char*>(address) && static_cast<const char*>(address) < end) {
        return &it->second.device_;
      }
      return std::nullopt;
    }

  private:
    struct AllocationInfo {
      size_t size_;
      const Device device_;

      AllocationInfo(size_t size, const Device& device) : size_(size), device_(device) {}
    };

    std::map<const void*, AllocationInfo> allocations_;
    std::mutex mutex_;
  };

} // namespace icicle