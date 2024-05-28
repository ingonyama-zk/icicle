#include "vec_ops/vec_ops.h"

using namespace icicle;

/*********************************** ADD ***********************************/
template <typename T>
class VectorAddDispatcher
{
public:
  static inline std::unordered_map<std::string /*device type*/, vectorOpImpl<T>> apiMap;

  static void _register(const std::string& deviceType, vectorOpImpl<T> func)
  {
    if (apiMap.find(deviceType) != apiMap.end()) {
      throw std::runtime_error(
        "Attempting to register a duplicate vector_add operation for device type: " + deviceType);
    }
    apiMap[deviceType] = func;
  }

  static eIcicleError execute(const T* vec_a, const T* vec_b, int n, const VecOpsConfig& config, T* output)
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
  return VectorAddDispatcher<scalar_t>::execute(vec_a, vec_b, n, config, output);
}

extern "C" void register_vector_add(const std::string& deviceType, vectorOpImpl<scalar_t> impl)
{
  std::cout << "vector_add registered for " << deviceType << std::endl;
  VectorAddDispatcher<scalar_t>::_register(deviceType, impl);
}

/*********************************** SUB ***********************************/
template <typename T>
class vectorSubDispatcher
{
public:
  static inline std::unordered_map<std::string /*device type*/, vectorOpImpl<T>> apiMap;

  static void _register(const std::string& deviceType, vectorOpImpl<T> func)
  {
    if (apiMap.find(deviceType) != apiMap.end()) {
      throw std::runtime_error(
        "Attempting to register a duplicate vector_add operation for device type: " + deviceType);
    }
    apiMap[deviceType] = func;
  }

  static eIcicleError execute(const T* vec_a, const T* vec_b, int n, const VecOpsConfig& config, T* output)
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

extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_sub)(
  const scalar_t* vec_a, const scalar_t* vec_b, int n, const VecOpsConfig& config, scalar_t* output)
{
  return vectorSubDispatcher<scalar_t>::execute(vec_a, vec_b, n, config, output);
}

extern "C" void register_vector_sub(const std::string& deviceType, vectorOpImpl<scalar_t> impl)
{
  std::cout << "vector_sub registered for " << deviceType << std::endl;
  vectorSubDispatcher<scalar_t>::_register(deviceType, impl);
}

/*********************************** MUL ***********************************/
template <typename T>
class vectorMulDispatcher
{
public:
  static inline std::unordered_map<std::string /*device type*/, vectorOpImpl<T>> apiMap;

  static void _register(const std::string& deviceType, vectorOpImpl<T> func)
  {
    if (apiMap.find(deviceType) != apiMap.end()) {
      throw std::runtime_error(
        "Attempting to register a duplicate vector_add operation for device type: " + deviceType);
    }
    apiMap[deviceType] = func;
  }

  static eIcicleError execute(const T* vec_a, const T* vec_b, int n, const VecOpsConfig& config, T* output)
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

extern "C" eIcicleError CONCAT_EXPAND(FIELD, vector_mul)(
  const scalar_t* vec_a, const scalar_t* vec_b, int n, const VecOpsConfig& config, scalar_t* output)
{
  return vectorMulDispatcher<scalar_t>::execute(vec_a, vec_b, n, config, output);
}

extern "C" void register_vector_mul(const std::string& deviceType, vectorOpImpl<scalar_t> impl)
{
  std::cout << "vector_mul registered for " << deviceType << std::endl;
  vectorMulDispatcher<scalar_t>::_register(deviceType, impl);
}