#include "vec_ops/vec_ops.h"

using namespace icicle;

template <typename T>
class vector_addDispatcher
{
public:
  static inline std::unordered_map<std::string /*device type*/, vector_addImpl<T>> apiMap;

  static void register_vector_add(const std::string& deviceType, vector_addImpl<T> func)
  {
    if (apiMap.find(deviceType) != apiMap.end()) {
      throw std::runtime_error(
        "Attempting to register a duplicate vector_add operation for device type: " + deviceType);
    }
    apiMap[deviceType] = func;
  }

  static eIcicleError executevector_add(const T* vec_a, const T* vec_b, int n, const VecOpsConfig& config, T* output)
  {
    const Device& device = DeviceAPI::get_thread_local_device();
    auto it = apiMap.find(device.type);
    if (it != apiMap.end()) {
      return it->second(device, vec_a, vec_b, n, config, output);
    } else {
      throw std::runtime_error("vector_add operation not supported on device " + std::string(device.type));
    }
  }
};

extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_add)(
  const scalar_t* vec_a, const scalar_t* vec_b, int n, const VecOpsConfig& config, scalar_t* output)
{
  return vector_addDispatcher<scalar_t>::executevector_add(vec_a, vec_b, n, config, output);
}

extern "C" void register_vector_add(const std::string& deviceType, vector_addImpl<scalar_t> impl)
{
  std::cout << "vectorAdd registered for " << deviceType << std::endl;
  vector_addDispatcher<scalar_t>::register_vector_add(deviceType, impl);
}
