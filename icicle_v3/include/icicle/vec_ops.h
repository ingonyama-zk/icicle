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
    bool is_b_on_device;       /**< True if `b` is on device and false if it is not. Default value: false. OPTIONAL*/
    bool is_result_on_device;  /**< If true, output is preserved on device, otherwise on host. Default value: false. */
    bool is_async; /**< Whether to run the vector operations asynchronously. If set to `true`, the function will be
                    *   non-blocking and you'd need to synchronize it explicitly by running
                    *   `cudaStreamSynchronize` or `cudaDeviceSynchronize`. If set to false, the
                    *   function will block the current CPU thread. */

    ConfigExtension* ext = nullptr; /** backend specific extension */
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

  // element wise vec ops
  template <typename T>
  eIcicleError vector_add(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  template <typename T>
  eIcicleError vector_sub(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  template <typename T>
  eIcicleError vector_mul(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  template <typename T>
  eIcicleError vector_div(const T* vec_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  template <typename T>
  eIcicleError convert_montgomery(const T* input, uint64_t size, bool is_into, const VecOpsConfig& config, T* output);

  // scalar-vec ops
  template <typename T>
  eIcicleError scalar_mul(const T* scalar_a, const T* vec_b, uint64_t size, const VecOpsConfig& config, T* output);

  // matrix ops
  template <typename T>
  eIcicleError
  matrix_transpose(const T* mat_in, uint32_t nof_rows, uint32_t nof_cols, const VecOpsConfig& config, T* mat_out);

  // misc
  template <typename T>
  eIcicleError bit_reverse(const T* vec_in, uint64_t size, const VecOpsConfig& config, T* vec_out);

  template <typename T>
  eIcicleError
  slice(const T* vec_in, uint64_t offset, uint64_t stride, uint64_t size, const VecOpsConfig& config, T* vec_out);

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

  void register_vector_div(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_VECTOR_DIV_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_div) = []() -> bool {                                                                  \
      register_vector_div(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_scalar_mul(const std::string& deviceType, scalarVectorOpImpl impl);

#define REGISTER_SCALAR_MUL_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_mul) = []() -> bool {                                                               \
      register_scalar_mul(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using scalarConvertMontgomeryImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* input,
    uint64_t size,
    bool is_into,
    const VecOpsConfig& config,
    scalar_t* output)>;

  void register_scalar_convert_montgomery(const std::string& deviceType, scalarConvertMontgomeryImpl);

#define REGISTER_CONVERT_MONTGOMERY_BACKEND(DEVICE_TYPE, FUNC)                                                         \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_convert_mont) = []() -> bool {                                                      \
      register_scalar_convert_montgomery(DEVICE_TYPE, FUNC);                                                           \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using scalarMatrixOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* in,
    uint32_t nof_rows,
    uint32_t nof_cols,
    const VecOpsConfig& config,
    scalar_t* out)>;

  void register_matrix_transpose(const std::string& deviceType, scalarMatrixOpImpl impl);

#define REGISTER_MATRIX_TRANSPOSE_BACKEND(DEVICE_TYPE, FUNC)                                                           \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_matrix_transpose) = []() -> bool {                                                         \
      register_matrix_transpose(DEVICE_TYPE, FUNC);                                                                    \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using scalarBitReverseOpImpl = std::function<eIcicleError(
    const Device& device, const scalar_t* input, uint64_t size, const VecOpsConfig& config, scalar_t* output)>;

  void register_scalar_bit_reverse(const std::string& deviceType, scalarBitReverseOpImpl);

#define REGISTER_BIT_REVERSE_BACKEND(DEVICE_TYPE, FUNC)                                                                \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_bit_reverse) = []() -> bool {                                                       \
      register_scalar_bit_reverse(DEVICE_TYPE, FUNC);                                                                  \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using scalarSliceOpImpl = std::function<eIcicleError(
    const Device& device,
    const scalar_t* input,
    uint64_t offset,
    uint64_t stride,
    uint64_t size,
    const VecOpsConfig& config,
    scalar_t* output)>;

  void register_slice(const std::string& deviceType, scalarSliceOpImpl);

#define REGISTER_SLICE_BACKEND(DEVICE_TYPE, FUNC)                                                                      \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_slice) = []() -> bool {                                                             \
      register_slice(DEVICE_TYPE, FUNC);                                                                               \
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

  void register_extension_vector_add(const std::string& deviceType, extFieldVectorOpImpl impl);

#define REGISTER_VECTOR_ADD_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                       \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_add_ext_field) = []() -> bool {                                                        \
      register_extension_vector_add(DEVICE_TYPE, FUNC);                                                                \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_extension_vector_sub(const std::string& deviceType, extFieldVectorOpImpl impl);
#define REGISTER_VECTOR_SUB_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                       \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_sub_ext_field) = []() -> bool {                                                        \
      register_extension_vector_sub(DEVICE_TYPE, FUNC);                                                                \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  void register_extension_vector_mul(const std::string& deviceType, extFieldVectorOpImpl impl);

#define REGISTER_VECTOR_MUL_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                       \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_vec_mul_ext_field) = []() -> bool {                                                        \
      register_extension_vector_mul(DEVICE_TYPE, FUNC);                                                                \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using extFieldConvertMontgomeryImpl = std::function<eIcicleError(
    const Device& device,
    const extension_t* input,
    uint64_t size,
    bool is_into,
    const VecOpsConfig& config,
    extension_t* output)>;

  extern "C" void
  register_extension_scalar_convert_montgomery(const std::string& deviceType, extFieldConvertMontgomeryImpl);

#define REGISTER_CONVERT_MONTGOMERY_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                               \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_convert_mont_ext_field) = []() -> bool {                                            \
      register_extension_scalar_convert_montgomery(DEVICE_TYPE, FUNC);                                                 \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using extFieldMatrixOpImpl = std::function<eIcicleError(
    const Device& device,
    const extension_t* in,
    uint32_t nof_rows,
    uint32_t nof_cols,
    const VecOpsConfig& config,
    extension_t* out)>;

  void register_extension_matrix_transpose(const std::string& deviceType, extFieldMatrixOpImpl impl);

#define REGISTER_MATRIX_TRANSPOSE_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_matrix_transpose_ext_field) = []() -> bool {                                               \
      register_extension_matrix_transpose(DEVICE_TYPE, FUNC);                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using extFieldBitReverseOpImpl = std::function<eIcicleError(
    const Device& device, const extension_t* input, uint64_t size, const VecOpsConfig& config, extension_t* output)>;

  void register_extension_bit_reverse(const std::string& deviceType, extFieldBitReverseOpImpl);

#define REGISTER_BIT_REVERSE_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                      \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_convert_mont) = []() -> bool {                                                      \
      register_extension_bit_reverse(DEVICE_TYPE, FUNC);                                                               \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  using extFieldSliceOpImpl = std::function<eIcicleError(
    const Device& device,
    const extension_t* input,
    uint64_t offset,
    uint64_t stride,
    uint64_t size,
    const VecOpsConfig& config,
    extension_t* output)>;

  void register_extension_slice(const std::string& deviceType, extFieldSliceOpImpl);

#define REGISTER_SLICE_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                            \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_scalar_slice) = []() -> bool {                                                             \
      register_extension_slice(DEVICE_TYPE, FUNC);                                                                     \
      return true;                                                                                                     \
    }();                                                                                                               \
  }
#endif // EXT_FIELD

} // namespace icicle