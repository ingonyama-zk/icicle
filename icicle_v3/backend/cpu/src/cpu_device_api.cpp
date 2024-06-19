
#include <iostream>
#include <cstring>
#include "icicle/device_api.h"
#include "icicle/errors.h"

using namespace icicle;

class CpuDeviceAPI : public DeviceAPI
{
public:
  eIcicleError set_device(const Device& device) override
  {
    return (device.id == 0) ? eIcicleError::SUCCESS : eIcicleError::INVALID_DEVICE;
  }

  // Memory management
  eIcicleError allocate_memory(void** ptr, size_t size) const override
  {
    *ptr = malloc(size);
    return (*ptr == nullptr) ? eIcicleError::ALLOCATION_FAILED : eIcicleError::SUCCESS;
  }

  eIcicleError allocate_memory_async(void** ptr, size_t size, icicleStreamHandle stream) const override
  {
    return CpuDeviceAPI::allocate_memory(ptr, size);
  }

  eIcicleError free_memory(void* ptr) const override
  {
    free(ptr);
    return eIcicleError::SUCCESS;
  }

  eIcicleError free_memory_async(void* ptr, icicleStreamHandle stream) const override
  {
    return CpuDeviceAPI::free_memory(ptr);
  }

  eIcicleError get_available_memory(size_t& total /*OUT*/, size_t& free /*OUT*/) const override
  {
    // TODO Yuval: implement this
    return eIcicleError::API_NOT_IMPLEMENTED;
  }

  eIcicleError memCopy(void* dst, const void* src, size_t size) const
  {
    std::memcpy(dst, src, size);
    return eIcicleError::SUCCESS;
  }

  // Data transfer
  eIcicleError copy_to_host(void* dst, const void* src, size_t size) const override { return memCopy(dst, src, size); }

  eIcicleError copy_to_host_async(void* dst, const void* src, size_t size, icicleStreamHandle stream) const override
  {
    return memCopy(dst, src, size);
  }

  eIcicleError copy_to_device(void* dst, const void* src, size_t size) const override
  {
    return memCopy(dst, src, size);
  }

  eIcicleError copy_to_device_async(void* dst, const void* src, size_t size, icicleStreamHandle stream) const override
  {
    return memCopy(dst, src, size);
  }

  // Synchronization
  eIcicleError synchronize(icicleStreamHandle stream = nullptr) const override { return eIcicleError::SUCCESS; }

  // Stream management
  eIcicleError create_stream(icicleStreamHandle* stream) const override
  {
    *stream = nullptr; // no streams for CPU
    return eIcicleError::SUCCESS;
  }

  eIcicleError destroy_stream(icicleStreamHandle stream) const override
  {
    return (nullptr == stream) ? eIcicleError::SUCCESS : eIcicleError::STREAM_DESTRUCTION_FAILED;
  }

  eIcicleError get_device_properties(DeviceProperties& properties) const override
  {
    properties.using_host_memory = true;
    properties.num_memory_regions = 0;
    properties.supports_pinned_memory = false;
    return eIcicleError::SUCCESS;
  }
};

REGISTER_DEVICE_API("CPU", CpuDeviceAPI);
REGISTER_DEVICE_API("CPU_REF", CpuDeviceAPI);
// class CpuRefDeviceAPI : public CpuDeviceAPI
// {};
// REGISTER_DEVICE_API("CPU_REF", CpuRefDeviceAPI);
