#pragma once

#include "icicle/utils/log.h"
#include "icicle/errors.h"
#include "icicle/device.h"
#include "icicle/device_api.h"
#include <unordered_map>
#include <typeinfo>
#include <cxxabi.h>
#include <memory>

using namespace icicle;

// Template function to demangle the name of the type
template <typename T>
std::string demangle()
{
  int status = -4;
  std::unique_ptr<char, void (*)(void*)> res{
    abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status), std::free};

  return (status == 0) ? res.get() : typeid(T).name();
}

// Generalized executor-dispatcher template
template <typename FuncType, const char* api_name>
class tIcicleExecuteDispatcher
{
  std::unordered_map<std::string /*device type*/, FuncType> apiMap;

public:
  static tIcicleExecuteDispatcher& Global()
  {
    static tIcicleExecuteDispatcher instance;
    return instance;
  }

  void _register(const std::string& deviceType, FuncType func)
  {
    if (apiMap.find(deviceType) != apiMap.end()) {
      THROW_ICICLE_ERR(
        eIcicleError::INVALID_DEVICE,
        std::string("Attempting to register a duplicate ") + api_name + " operation for device type: " + deviceType);
    }
    apiMap[deviceType] = func;
  }

  template <typename... Args>
  static auto execute(Args&&... args) -> decltype(auto)
  {
    const Device& device = DeviceAPI::get_thread_local_device();
    auto& apiMap = Global().apiMap;
    auto it = apiMap.find(device.type);
    if (it != apiMap.end()) {
      return it->second(device, std::forward<Args>(args)...);
    } else {
      THROW_ICICLE_ERR(
        eIcicleError::INVALID_DEVICE, std::string(api_name) + " operation not supported on device " + device.type);
    }
    return eIcicleError::SUCCESS;
  }
};

// Instantiate a dispatcher class and the corresponding registration function
#define ICICLE_DISPATCHER_INST(dispatcher_class_name, api_name, type)                                                  \
  constexpr char ST_name_##api_name[]{#api_name};                                                                      \
  using dispatcher_class_name = tIcicleExecuteDispatcher<type, ST_name_##api_name>;                                    \
  void register_##api_name(const std::string& deviceType, type impl)                                                   \
  {                                                                                                                    \
    ICICLE_LOG_DEBUG << " Registering API: device=" << deviceType << ", api=" << #api_name << "<" << demangle<type>()  \
                     << ">";                                                                                           \
    dispatcher_class_name::Global()._register(deviceType, impl);                                                       \
  }

/********************************************************************************/

// Generalized constructor-dispatcher type template
template <typename FactoryType, const char* api_name>
class tIcicleObjectDispatcher
{
  std::unordered_map<std::string /*device type*/, std::shared_ptr<FactoryType>> apiMap;

public:
  static tIcicleObjectDispatcher& Global()
  {
    static tIcicleObjectDispatcher instance;
    return instance;
  }

  void _register(const std::string& deviceType, std::shared_ptr<FactoryType> factory)
  {
    if (apiMap.find(deviceType) != apiMap.end()) {
      THROW_ICICLE_ERR(
        eIcicleError::INVALID_DEVICE,
        std::string("Attempting to register a duplicate ") + api_name + " operation for device type: " + deviceType);
    }
    apiMap[deviceType] = factory;
  }

  static const std::shared_ptr<FactoryType> get_factory()
  {
    const Device& device = DeviceAPI::get_thread_local_device();
    auto& apiMap = Global().apiMap;
    auto it = apiMap.find(device.type);
    if (it != apiMap.end()) {
      return it->second;
    } else {
      THROW_ICICLE_ERR(eIcicleError::INVALID_DEVICE, std::string(api_name) + " not supported on device " + device.type);
    }
    return nullptr;
  }
};

// Instantiate a dispatcher class and the corresponding registration function
#define ICICLE_OBJECT_DISPATCHER_INST(dispatcher_class_name, api_name, type)                                           \
  constexpr char ST_name_##api_name[]{#api_name};                                                                      \
  using dispatcher_class_name = tIcicleObjectDispatcher<type, ST_name_##api_name>;                                     \
  void register_##api_name(const std::string& deviceType, std::shared_ptr<type> factory)                               \
  {                                                                                                                    \
    ICICLE_LOG_DEBUG << " Registering API: device=" << deviceType << ", api=" << #api_name << "<"                      \
                     << demangle<type> << ">";                                                                         \
    dispatcher_class_name::Global()._register(deviceType, factory);                                                    \
  }
