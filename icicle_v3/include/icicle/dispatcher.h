#pragma once

// Generalized dispatcher template
template <typename FuncType, const char* api_name>
class tIcicleDispatcher
{
public:
  static inline std::unordered_map<std::string /*device type*/, FuncType> apiMap;

  static void _register(const std::string& deviceType, FuncType func)
  {
    if (apiMap.find(deviceType) != apiMap.end()) {
      throw std::runtime_error(
        std::string("Attempting to register a duplicate ") + api_name + " operation for device type: " + deviceType);
    }
    apiMap[deviceType] = func;
  }

  template <typename... Args>
  static auto execute(Args... args) -> decltype(auto)
  {
    const Device& device = DeviceAPI::get_thread_local_device();
    auto it = apiMap.find(device.type);
    if (it != apiMap.end()) {
      return it->second(device, args...);
    } else {
      throw std::runtime_error(std::string(api_name) + " operation not supported on device " + device.type);
    }
  }
};

// Instantiate a dispatcher class and the corresponding registration function
#define ICICLE_DISPATCHER_INST(dispatcher_class_name, api_name, type)                                                  \
  constexpr char ST_name_##api_name[]{#api_name};                                                                      \
  using dispatcher_class_name = tIcicleDispatcher<type, ST_name_##api_name>;                                           \
  extern "C" void register_##api_name(const std::string& deviceType, type impl)                                        \
  {                                                                                                                    \
    std::cout << #api_name << " registered for " << deviceType << std::endl;                                           \
    dispatcher_class_name::_register(deviceType, impl);                                                                \
  }
