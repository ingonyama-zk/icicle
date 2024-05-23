
#include <iostream>
#include <cstring>
#include "icicle/device_api.h"
#include "icicle/errors.h"

using namespace icicle;

class CPUDeviceAPI : public DeviceAPI
{
public:
  // Memory management
  eIcicleError allocateMemory(const Device& device, void** ptr, size_t size) override
  {
    if (device.id != 0) return eIcicleError::INVALID_DEVICE;
    *ptr = malloc(size);
    return (*ptr == nullptr) ? eIcicleError::ALLOCATION_FAILED : eIcicleError::SUCCESS;
  }

  eIcicleError allocateMemoryAsync(const Device& device, void** ptr, size_t size, IcicleStreamHandle stream) override
  {
    if (device.id != 0) return eIcicleError::INVALID_DEVICE;
    return CPUDeviceAPI::allocateMemory(device, ptr, size);
  }

  eIcicleError freeMemory(const Device& device, void* ptr) override
  {
    if (device.id != 0) return eIcicleError::INVALID_DEVICE;
    free(ptr);
    return eIcicleError::SUCCESS;
  }

  eIcicleError freeMemoryAsync(const Device& device, void* ptr, IcicleStreamHandle stream) override
  {
    if (device.id != 0) return eIcicleError::INVALID_DEVICE;
    return CPUDeviceAPI::freeMemory(device, ptr);
  }

  eIcicleError getAvailableMemory(const Device& device, size_t& total /*OUT*/, size_t& free /*OUT*/) override
  {
    if (device.id != 0) return eIcicleError::INVALID_DEVICE;
    // TODO Yuval: implement this
    return eIcicleError::API_NOT_IMPLEMENTED;
  }

  eIcicleError memCopy(void* dst, const void* src, size_t size)
  {
    std::memcpy(dst, src, size);
    return eIcicleError::SUCCESS;
  }

  // Data transfer
  eIcicleError copyToHost(const Device& device, void* dst, const void* src, size_t size) override
  {
    if (device.id != 0) return eIcicleError::INVALID_DEVICE;
    return memCopy(dst, src, size);
  }

  eIcicleError
  copyToHostAsync(const Device& device, void* dst, const void* src, size_t size, IcicleStreamHandle stream) override
  {
    if (device.id != 0) return eIcicleError::INVALID_DEVICE;
    return memCopy(dst, src, size);
  }

  eIcicleError copyToDevice(const Device& device, void* dst, const void* src, size_t size) override
  {
    if (device.id != 0) return eIcicleError::INVALID_DEVICE;
    return memCopy(dst, src, size);
  }

  eIcicleError
  copyToDeviceAsync(const Device& device, void* dst, const void* src, size_t size, IcicleStreamHandle stream) override
  {
    if (device.id != 0) return eIcicleError::INVALID_DEVICE;
    return memCopy(dst, src, size);
  }

  // Synchronization
  eIcicleError synchronize(const Device& device, IcicleStreamHandle stream = nullptr) override
  {
    if (device.id != 0) return eIcicleError::INVALID_DEVICE;
    return eIcicleError::SUCCESS;
  }

  // Stream management
  eIcicleError createStream(const Device& device, IcicleStreamHandle* stream) override
  {
    if (device.id != 0) return eIcicleError::INVALID_DEVICE;
    *stream = nullptr; // no streams for CPU
    return eIcicleError::SUCCESS;
  }

  eIcicleError destroyStream(const Device& device, IcicleStreamHandle stream) override
  {
    if (device.id != 0) return eIcicleError::INVALID_DEVICE;
    return (nullptr == stream) ? eIcicleError::SUCCESS : eIcicleError::STREAM_DESTRUCTION_FAILED;
  }
};

REGISTER_DEVICE_API("CPU", CPUDeviceAPI);
