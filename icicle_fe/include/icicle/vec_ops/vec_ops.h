#pragma once

#include <functional>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <iostream>

#include "errors.h"
#include "device.h"

#include "fields/field.h"
#include "fields/field_config.h"

using namespace field_config;

namespace icicle {

  // Template alias for a function implementing vector addition for a specific device and type T
  template <typename T>
  using VectorAddImpl =
    std::function<IcicleError(const Device& device, const T* vec_a, const T* vec_b, int n, T* output)>;

  // Declaration of the vector addition function for integer vectors
  // This function performs element-wise addition of two integer vectors on a specified device
  extern "C" IcicleError
  VectorAdd(const Device& device, const scalar_t* vec_a, const scalar_t* vec_b, int n, scalar_t* output);

  // Function to register a vector addition implementation for a specific device type
  // This allows the system to use the appropriate implementation based on the device type
  extern "C" void registerVectorAdd(const std::string& deviceType, VectorAddImpl<scalar_t> impl);

// Macro to simplify the registration of a vector addition implementation for a device type
// Usage: REGISTER_VECTOR_ADD_BACKEND("device_type", implementation_function)
#define REGISTER_VECTOR_ADD_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool _reg_vec_add = []() -> bool {                                                                          \
      registerVectorAdd(DEVICE_TYPE, FUNC);                                                                            \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

} // namespace icicle