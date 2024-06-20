
#include <memory>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <iostream>

#include "icicle/runtime.h"
#include "icicle/device_api.h"
#include "icicle/utils/log.h"

namespace icicle {

  class DeviceAPIRegistry
  {
    std::unordered_map<std::string, std::shared_ptr<DeviceAPI>> apiMap;

  public:
    static DeviceAPIRegistry& Global()
    {
      static DeviceAPIRegistry instance;
      return instance;
    }

    void register_deviceAPI(const std::string& deviceType, std::shared_ptr<DeviceAPI> api)
    {
      if (apiMap.find(deviceType) != apiMap.end()) {
        throw std::runtime_error("Attempting to register a duplicate API for device type: " + deviceType);
      }
      const bool is_first = apiMap.empty();
      apiMap[deviceType] = api;
      if (is_first) {
        Device dev{deviceType, 0};
        m_default_device = dev;
        api->set_device(dev);
      }
    }

    std::shared_ptr<DeviceAPI> get_deviceAPI(const Device& device)
    {
      auto it = apiMap.find(device.type);
      if (it != apiMap.end()) {
        return it->second;
      } else {
        throw std::runtime_error("Device API not found for type: " + std::string(device.type));
      }
    }

    std::shared_ptr<DeviceAPI> get_default_deviceAPI()
    {
      const bool have_default_device = m_default_device.id >= 0;
      if (!have_default_device) { return nullptr; }
      return get_deviceAPI(m_default_device);
    }

    const Device& get_default_device() { return m_default_device; }

    std::list<std::string> get_registered_devices()
    {
      std::list<std::string> registered_devices;
      for (const auto& device : apiMap) {
        registered_devices.push_back(device.first);
      }
      return registered_devices;
    }

    bool is_device_registered(const char* device_type) { return apiMap.find(device_type) != apiMap.end(); }

    Device m_default_device{"", -1};
  };

  thread_local Device DeviceAPI::sCurDevice = {"", -1};
  thread_local const DeviceAPI* DeviceAPI::sCurDeviceAPI = nullptr;

  eIcicleError DeviceAPI::set_thread_local_device(const Device& device)
  {
    DeviceAPI* device_api = get_deviceAPI(device);
    if (nullptr == device_api) return eIcicleError::INVALID_DEVICE;

    auto err = device_api->set_device(device); // notifying the device backend about the device set

    if (err == eIcicleError::SUCCESS) {
      sCurDevice = device;
      sCurDeviceAPI = device_api;
    }
    return err;
  }

  const Device& DeviceAPI::get_thread_local_device()
  {
    const bool is_device_set = sCurDevice.id >= 0;
    if (is_device_set) { return sCurDevice; }

    const Device& default_device = DeviceAPIRegistry::Global().get_default_device();
    if (default_device.id < 0) {
      throw std::runtime_error("icicle Device is not set. Make sure to load and initialize a device via call to "
                               "icicle_set_device(const icicle::Device& device)");
    }
    return default_device;
  }

  const DeviceAPI* DeviceAPI::get_thread_local_deviceAPI()
  {
    if (nullptr != sCurDeviceAPI) { return sCurDeviceAPI; }
    auto default_deviceAPI = DeviceAPIRegistry::Global().get_default_deviceAPI();
    if (nullptr == default_deviceAPI) {
      throw std::runtime_error("icicle Device is not set. Make sure to load and initialize a device via call to "
                               "icicle_set_device(const icicle::Device& device)");
    }
    return default_deviceAPI.get();
  }

  DeviceAPI* get_deviceAPI(const Device& device) { return DeviceAPIRegistry::Global().get_deviceAPI(device).get(); }

  extern "C" void register_deviceAPI(const std::string& deviceType, std::shared_ptr<DeviceAPI> api)
  {
    ICICLE_LOG_DEBUG << "deviceAPI registered for " << deviceType;
    DeviceAPIRegistry::Global().register_deviceAPI(deviceType, api);
  }

  std::list<std::string> get_registered_devices() { return DeviceAPIRegistry::Global().get_registered_devices(); }

  bool is_device_registered(const char* device_type)
  {
    return DeviceAPIRegistry::Global().is_device_registered(device_type);
  }

} // namespace icicle