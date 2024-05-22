#include "vec_ops/vec_ops.h"

using namespace icicle;

template <typename T>
class VectorAddDispatcher
{
public:
  static inline std::unordered_map<std::string /*device type*/, VectorAddImpl<T>> apiMap;

  static void registerVectorAdd(const std::string& deviceType, VectorAddImpl<T> func)
  {
    if (apiMap.find(deviceType) != apiMap.end()) {
      throw std::runtime_error("Attempting to register a duplicate VectorAdd operation for device type: " + deviceType);
    }
    apiMap[deviceType] = func;
  }

  static IcicleError executeVectorAdd(const Device& device, const T* vec_a, const T* vec_b, int n, T* output)
  {
    auto it = apiMap.find(device.type);
    if (it != apiMap.end()) {
      return it->second(device, vec_a, vec_b, n, output);
    } else {
      throw std::runtime_error("VectorAdd operation not supported on device " + std::string(device.type));
    }
  }
};

extern "C" IcicleError
VectorAdd(const Device& device, const scalar_t* vec_a, const scalar_t* vec_b, int n, scalar_t* output)
{
  return VectorAddDispatcher<scalar_t>::executeVectorAdd(device, vec_a, vec_b, n, output);
}

extern "C" void registerVectorAdd(const std::string& deviceType, VectorAddImpl<scalar_t> impl)
{
  std::cout << "vectorAdd registered for " << deviceType << std::endl;
  VectorAddDispatcher<scalar_t>::registerVectorAdd(deviceType, impl);
}
