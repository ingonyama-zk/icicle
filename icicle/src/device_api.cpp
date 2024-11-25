
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>

#include "icicle/runtime.h"
#include "icicle/device_api.h"
#include "icicle/errors.h"
#include "icicle/utils/log.h"
#include "icicle/memory_tracker.h"

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
        ICICLE_LOG_ERROR << "Attempting to register a duplicate API for device type: " << deviceType << ". Skipping";
        return;
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
      if (it == apiMap.end()) {
        THROW_ICICLE_ERR(eIcicleError::INVALID_DEVICE, "Device API not found for type: " + std::string(device.type));
      }
      return it->second;
    }

    std::shared_ptr<DeviceAPI> get_default_deviceAPI()
    {
      const bool have_default_device = m_default_device.id >= 0;
      if (!have_default_device) { return nullptr; }
      return get_deviceAPI(m_default_device);
    }

    const Device& get_default_device() { return m_default_device; }

    std::vector<std::string> get_registered_devices_list()
    {
      std::vector<std::string> registered_devices;
      for (const auto& device : apiMap) {
        registered_devices.push_back(device.first);
      }
      return registered_devices;
    }

    bool is_device_registered(const char* device_type) { return apiMap.find(device_type) != apiMap.end(); }

    Device m_default_device{"", -1};
  };

  DeviceTracker DeviceAPI::sMemTracker;
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
      THROW_ICICLE_ERR(
        eIcicleError::INVALID_DEVICE, "icicle Device is not set. Make sure to load and initialize a device via call to "
                                      "icicle_set_device(const icicle::Device& device)");
    }
    return default_device;
  }

  const DeviceAPI* DeviceAPI::get_thread_local_deviceAPI()
  {
    if (nullptr != sCurDeviceAPI) { return sCurDeviceAPI; }
    auto default_deviceAPI = DeviceAPIRegistry::Global().get_default_deviceAPI();
    if (nullptr == default_deviceAPI) {
      THROW_ICICLE_ERR(
        eIcicleError::INVALID_DEVICE, "icicle Device is not set. Make sure to load and initialize a device via call to "
                                      "icicle_set_device(const icicle::Device& device)");
    }
    return default_deviceAPI.get();
  }

  /********************************************************************************** */

  DeviceAPI* get_deviceAPI(const Device& device) { return DeviceAPIRegistry::Global().get_deviceAPI(device).get(); }

  void register_deviceAPI(const std::string& deviceType, std::shared_ptr<DeviceAPI> api)
  {
    ICICLE_LOG_DEBUG << " Registering DEVICE: device=" << deviceType;
    DeviceAPIRegistry::Global().register_deviceAPI(deviceType, api);
  }

  std::vector<std::string> get_registered_devices_list()
  {
    return DeviceAPIRegistry::Global().get_registered_devices_list();
  }

  bool is_device_registered(const char* device_type)
  {
    return DeviceAPIRegistry::Global().is_device_registered(device_type);
  }

  eIcicleError get_registered_devices(char* output, size_t output_size)
  {
    if (output == nullptr) { return eIcicleError::INVALID_POINTER; }

    std::vector<std::string> devices = get_registered_devices_list();
    std::string concatenated_devices;

    for (const auto& device : devices) {
      if (!concatenated_devices.empty()) { concatenated_devices += ","; }
      concatenated_devices += device;
    }

    if (concatenated_devices.size() + 1 > output_size) { // +1 for null-terminator
      return eIcicleError::OUT_OF_MEMORY;
    }

    std::strncpy(output, concatenated_devices.c_str(), output_size - 1);
    output[output_size - 1] = '\0'; // Ensure null-termination

    return eIcicleError::SUCCESS;
  }

} // namespace icicle