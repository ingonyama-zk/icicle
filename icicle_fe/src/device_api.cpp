
#include <memory>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <iostream>

#include "runtime.h"
#include "device_api.h"

using namespace icicle;

eIcicleError DeviceAPI::setThreadLocalDevice(const Device& device)
{
  DeviceAPI* device_api = getDeviceAPI(device);
  if (nullptr == device_api) return eIcicleError::INVALID_DEVICE;

  auto err = device_api->setDevice(device); // notifying the device backend about the device set
  if (err == eIcicleError::SUCCESS) {
    sCurDevice = device;
    sCurDeviceAPI = device_api;
  }
  return err;
}

const Device& DeviceAPI::getThreadLocalDevice()
{
  const bool is_valid_device = sCurDevice.id >= 0 && sCurDevice.type != nullptr;
  if (!is_valid_device) {
    throw std::runtime_error("icicle Device is not set. Make sure to initialize the device via call to "
                             "icicleSetDevice(const icicle::Device& device)");
  }
  return sCurDevice;
}
const DeviceAPI* DeviceAPI::getThreadLocalDeviceAPI() { return sCurDeviceAPI; }

class DeviceAPIRegistry
{
  static inline std::unordered_map<std::string, std::shared_ptr<DeviceAPI>> apiMap;

public:
  static void registerDeviceAPI(const std::string& deviceType, std::shared_ptr<DeviceAPI> api)
  {
    if (apiMap.find(deviceType) != apiMap.end()) {
      throw std::runtime_error("Attempting to register a duplicate API for device type: " + deviceType);
    }
    apiMap[deviceType] = api;
  }

  static std::shared_ptr<DeviceAPI> getDeviceAPI(const Device& device)
  {
    auto it = apiMap.find(device.type);
    if (it != apiMap.end()) {
      return it->second;
    } else {
      throw std::runtime_error("Device API not found for type: " + std::string(device.type));
    }
  }

  static std::list<std::string> getRegisteredDevices()
  {
    std::list<std::string> registered_devices;
    for (const auto& device : apiMap) {
      registered_devices.push_back(device.first);
    }
    return registered_devices;
  }
};

extern "C" DeviceAPI* getDeviceAPI(const Device& device) { return DeviceAPIRegistry::getDeviceAPI(device).get(); }

extern "C" void registerDeviceAPI(const std::string& deviceType, std::shared_ptr<DeviceAPI> api)
{
  std::cout << "deviceAPI registered for " << deviceType << std::endl;
  DeviceAPIRegistry::registerDeviceAPI(deviceType, api);
}

extern "C" std::list<std::string> getRegisteredDevices() { return DeviceAPIRegistry::getRegisteredDevices(); }
