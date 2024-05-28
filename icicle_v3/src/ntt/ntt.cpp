#include "ntt/ntt.h"

using namespace icicle;

#include "vec_ops/vec_ops.h"

using namespace icicle;

template <typename S, typename E>
class NttDispatcher
{
public:
  static inline std::unordered_map<std::string /*device type*/, NttImpl<S, E>> apiMap;

  static void register_ntt(const std::string& deviceType, NttImpl<S, E> func)
  {
    if (apiMap.find(deviceType) != apiMap.end()) {
      throw std::runtime_error("Attempting to register a duplicate Ntt operation for device type: " + deviceType);
    }
    apiMap[deviceType] = func;
  }

  static eIcicleError executeNtt(const E* input, int size, NTTDir dir, NTTConfig<S>& config, E* output)
  {
    const Device& device = DeviceAPI::get_thread_local_device();
    auto it = apiMap.find(device.type);
    if (it != apiMap.end()) {
      return it->second(device, input, size, dir, config, output);
    } else {
      throw std::runtime_error("Ntt operation not supported on device " + std::string(device.type));
    }
  }
};

extern "C" eIcicleError
CONCAT_EXPAND(FIELD, ntt)(const scalar_t* input, int size, NTTDir dir, NTTConfig<scalar_t>& config, scalar_t* output)
{
  return NttDispatcher<scalar_t, scalar_t>::executeNtt(input, size, dir, config, output);
}

extern "C" void register_ntt(const std::string& deviceType, NttImpl<scalar_t, scalar_t> impl)
{
  std::cout << "Ntt registered for " << deviceType << std::endl;
  NttDispatcher<scalar_t, scalar_t>::register_ntt(deviceType, impl);
}
