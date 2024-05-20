
#include <iostream>
#include "icicle/device_api.h"
#include "icicle/errors.h"

using namespace icicle;

class CPUDeviceAPI : public DeviceAPI
{
public:
  // Memory management
  IcicleError allocateMemory(const Device& device, void** ptr, size_t size) override
  {
    std::cout << "CPU" << std::endl;
    return IcicleError::SUCCESS;
  }

  IcicleError allocateMemoryAsync(const Device& device, void** ptr, size_t size, IcicleStream* stream) override
  {
    std::cout << "CPU" << std::endl;
    return IcicleError::SUCCESS;
  }

  IcicleError freeMemory(const Device& device, void* ptr) override
  {
    std::cout << "CPU" << std::endl;
    return IcicleError::SUCCESS;
  }

  IcicleError freeMemoryAsync(const Device& device, void* ptr, IcicleStream* stream) override
  {
    std::cout << "CPU" << std::endl;
    return IcicleError::SUCCESS;
  }
  IcicleError getAvailableMemory(const Device& device, size_t& total /*OUT*/, size_t& free /*OUT*/) override
  {
    std::cout << "CPU" << std::endl;
    return IcicleError::SUCCESS;
  }

  // Data transfer
  IcicleError copyToHost(const Device& device, void* dst, const void* src, size_t size) override
  {
    std::cout << "CPU" << std::endl;
    return IcicleError::SUCCESS;
  }
  IcicleError
  copyToHostAsync(const Device& device, void* dst, const void* src, size_t size, IcicleStream* stream) override
  {
    std::cout << "CPU" << std::endl;
    return IcicleError::SUCCESS;
  }

  IcicleError copyToDevice(const Device& device, void* dst, const void* src, size_t size) override
  {
    std::cout << "CPU" << std::endl;
    return IcicleError::SUCCESS;
  }

  IcicleError
  copyToDeviceAsync(const Device& device, void* dst, const void* src, size_t size, IcicleStream* stream) override
  {
    std::cout << "CPU" << std::endl;
    return IcicleError::SUCCESS;
  }

  // Synchronization
  IcicleError synchronize(const Device& device, IcicleStream* stream = nullptr) override
  {
    std::cout << "CPU" << std::endl;
    return IcicleError::SUCCESS;
  }

  // Stream management
  IcicleError createStream(const Device& device, IcicleStream** stream) override
  {
    std::cout << "CPU" << std::endl;
    return IcicleError::SUCCESS;
  }

  IcicleError destroyStream(const Device& device, IcicleStream* stream) override
  {
    std::cout << "CPU" << std::endl;
    return IcicleError::SUCCESS;
  }
};

REGISTER_DEVICE_API("CPU", CPUDeviceAPI);
