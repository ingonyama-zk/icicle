
#include <iostream>
#include <cstring>
#include "icicle/device_api.h"
#include "icicle/errors.h"

using namespace icicle;

class CPUDeviceAPI : public DeviceAPI
{
public:
  eIcicleError setDevice(const Device& device) override
  {
    return (device.id == 0) ? eIcicleError::SUCCESS : eIcicleError::INVALID_DEVICE;
  }

  // Memory management
  eIcicleError allocateMemory(void** ptr, size_t size) const override
  {
    *ptr = malloc(size);
    return (*ptr == nullptr) ? eIcicleError::ALLOCATION_FAILED : eIcicleError::SUCCESS;
  }

  eIcicleError allocateMemoryAsync(void** ptr, size_t size, icicleStreamHandle stream) const override
  {
    return CPUDeviceAPI::allocateMemory(ptr, size);
  }

  eIcicleError freeMemory(void* ptr) const override
  {
    free(ptr);
    return eIcicleError::SUCCESS;
  }

  eIcicleError freeMemoryAsync(void* ptr, icicleStreamHandle stream) const override
  {
    return CPUDeviceAPI::freeMemory(ptr);
  }

  eIcicleError getAvailableMemory(size_t& total /*OUT*/, size_t& free /*OUT*/) const override
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
  eIcicleError copyToHost(void* dst, const void* src, size_t size) const override { return memCopy(dst, src, size); }

  eIcicleError copyToHostAsync(void* dst, const void* src, size_t size, icicleStreamHandle stream) const override
  {
    return memCopy(dst, src, size);
  }

  eIcicleError copyToDevice(void* dst, const void* src, size_t size) const override { return memCopy(dst, src, size); }

  eIcicleError copyToDeviceAsync(void* dst, const void* src, size_t size, icicleStreamHandle stream) const override
  {
    return memCopy(dst, src, size);
  }

  // Synchronization
  eIcicleError synchronize(icicleStreamHandle stream = nullptr) const override { return eIcicleError::SUCCESS; }

  // Stream management
  eIcicleError createStream(icicleStreamHandle* stream) const override
  {
    *stream = nullptr; // no streams for CPU
    return eIcicleError::SUCCESS;
  }

  eIcicleError destroyStream(icicleStreamHandle stream) const override
  {
    return (nullptr == stream) ? eIcicleError::SUCCESS : eIcicleError::STREAM_DESTRUCTION_FAILED;
  }
};

REGISTER_DEVICE_API("CPU", CPUDeviceAPI);
