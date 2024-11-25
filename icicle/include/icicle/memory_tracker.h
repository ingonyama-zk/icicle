#pragma once

#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <thread>

namespace icicle {

  template <typename T>
  class MemoryTracker
  {
  public:
    // Add a new allocation with a void* address and an associated data of type T
    void add_allocation(const void* address, size_t size, T associated_data)
    {
      std::lock_guard<std::mutex> lock(mutex_);
      allocations_.insert(std::make_pair(address, AllocationInfo{size, associated_data}));
    }

    // Remove an allocation
    void remove_allocation(const void* address)
    {
      std::lock_guard<std::mutex> lock(mutex_);
      allocations_.erase(address);
    }

    // Identify the base address and offset for a given address
    std::optional<std::pair<const T*, size_t /*offset*/>> identify(const void* address)
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = allocations_.upper_bound(address);
      if (it == allocations_.begin()) { return std::nullopt; }
      --it;
      const char* start = static_cast<const char*>(it->first);
      const char* end = start + it->second.size_;
      if (start <= static_cast<const char*>(address) && static_cast<const char*>(address) < end) {
        size_t offset = static_cast<const char*>(address) - start;
        return std::make_pair(&it->second.associated_data_, offset);
      }
      return std::nullopt;
    }

  private:
    struct AllocationInfo {
      size_t size_;
      const T associated_data_;

      AllocationInfo(size_t size, T associated_data) : size_{size}, associated_data_{associated_data} {}
    };

    std::map<const void*, AllocationInfo> allocations_;
    std::mutex mutex_;
  };

} // namespace icicle