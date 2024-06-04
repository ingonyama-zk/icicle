#pragma once

#include <functional>

#include "errors.h"
#include "runtime.h"

#include "icicle/fields/field.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"
#include "icicle/config_extension.h"

using namespace field_config;

namespace icicle {

  /*************************** Frontend APIs ***************************/
  struct VecOpsConfig {
    icicleStreamHandle stream; /**< stream for async execution. */
    bool is_a_on_device;       /**< True if `a` is on device and false if it is not. Default value: false. */
    bool is_b_on_device;       /**< True if `b` is on device and false if it is not. Default value: false. */
    bool is_result_on_device;  /**< If true, output is preserved on device, otherwise on host. Default value: false. */
    bool is_async; /**< Whether to run the vector operations asynchronously. If set to `true`, the function will be
                    *   non-blocking and you'd need to synchronize it explicitly by running
                    *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the
                    *   function will block the current CPU thread. */

    ConfigExtension ext; /** backend specific extensions*/
  };

  /**
   * A function that returns the default value of [VecOpsConfig](@ref VecOpsConfig).
   * @return Default value of [VecOpsConfig](@ref VecOpsConfig).
   */
  static VecOpsConfig default_vec_ops_config()
  {
    VecOpsConfig config = {
      nullptr, // stream
      false,   // is_a_on_device
      false,   // is_b_on_device
      false,   // is_result_on_device
      false,   // is_async
    };
    return config;
  }

  // template APIs

  template <typename S>
  eIcicleError vector_add(const S* vec_a, const S* vec_b, uint64_t n, const VecOpsConfig& config, S* output);

  template <typename S>
  eIcicleError vector_sub(const S* vec_a, const S* vec_b, uint64_t n, const VecOpsConfig& config, S* output);

  template <typename S>
  eIcicleError vector_mul(const S* vec_a, const S* vec_b, uint64_t n, const VecOpsConfig& config, S* output);

  template <typename S>
  eIcicleError scalar_convert_montgomery(S* scalars, uint64_t size, bool is_into, const VecOpsConfig& config);

  /*************************** Backend registration ***************************/

  using scalarVectorOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* vec_a,
    const scalar_t* vec_b,
    uint64_t n,
    const VecOpsConfig& config,
    scalar_t* output)>;

  void register_vector_add(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_VECTOR_ADD_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_add) = []() -> bool {                                                                  \
      register_vector_add(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_sub(const std::string& deviceType, scalarVectorOpImpl impl);
#define REGISTER_VECTOR_SUB_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_sub) = []() -> bool {                                                                  \
      register_vector_sub(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_mul(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_VECTOR_MUL_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_mul) = []() -> bool {                                                                  \
      register_vector_mul(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using scalarConvertMontgomeryImpl = std::function<eIcicleError(
    const Device& device, scalar_t* scalars, uint64_t size, bool is_into, const VecOpsConfig& config)>;

  void register_scalar_convert_montgomery(const std::string& deviceType, scalarConvertMontgomeryImpl);

#define REGISTER_CONVERT_MONTGOMERY_BACKEND(DEVICE_TYPE, FUNC)                                                         \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_convert_mont) = []() -> bool {                                                      \
      register_scalar_convert_montgomery(DEVICE_TYPE, FUNC);                                                           \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

#ifdef EXT_FIELD
  using extFieldVectorOpImpl = std::function<eIcicleError(
    const Device& device,
    const extension_t* vec_a,
    const extension_t* vec_b,
    uint64_t n,
    const VecOpsConfig& config,
    extension_t* output)>;

  void register_vector_add_ext_field(const std::string& deviceType, extFieldVectorOpImpl impl);

#define REGISTER_VECTOR_ADD_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                       \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_add_ext_field) = []() -> bool {                                                        \
      register_vector_add_ext_field(DEVICE_TYPE, FUNC);                                                                \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_sub_ext_field(const std::string& deviceType, extFieldVectorOpImpl impl);
#define REGISTER_VECTOR_SUB_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                       \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_sub_ext_field) = []() -> bool {                                                        \
      register_vector_sub_ext_field(DEVICE_TYPE, FUNC);                                                                \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_vector_mul_ext_field(const std::string& deviceType, extFieldVectorOpImpl impl);

#define REGISTER_VECTOR_MUL_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                       \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_mul_ext_field) = []() -> bool {                                                        \
      register_vector_mul_ext_field(DEVICE_TYPE, FUNC);                                                                \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using extFieldConvertMontgomeryImpl = std::function<eIcicleError(
    const Device& device, extension_t* scalars, uint64_t size, bool is_into, const VecOpsConfig& config)>;

  void register_scalar_convert_montgomery_ext_field(const std::string& deviceType, extFieldConvertMontgomeryImpl);

#define REGISTER_CONVERT_MONTGOMERY_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                               \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_convert_mont_ext_field) = []() -> bool {                                            \
      register_scalar_convert_montgomery_ext_field(DEVICE_TYPE, FUNC);                                                 \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

#endif // EXT_FIELD

} // namespace icicle