#pragma once

#include <functional>

#include "icicle/errors.h"
#include "icicle/runtime.h"

#include "icicle/fields/field.h"
#include "icicle/fields/field_config.h"
#include "icicle/utils/utils.h"
#include "icicle/config_extension.h"

using namespace field_config;

namespace icicle {

  /*************************** Frontend APIs ***************************/

  /**
   * @enum NTTDir
   * @brief Specifies whether to perform forward NTT or inverse NTT (iNTT).
   *
   * Forward NTT computes polynomial evaluations from coefficients while inverse NTT computes coefficients from
   * evaluations.
   */
  enum class NTTDir {
    kForward, /**< Perform forward NTT. */
    kInverse  /**< Perform inverse NTT (iNTT). */
  };

  /**
   * @enum Ordering
   * @brief Specifies the ordering of inputs and outputs for the NTT.
   *
   * Note: kNM, kMN are efficient when using the Mixed-Radix NTT algorithm. For Radix2, kNM==kNR, kMN==kRN.
   * M stands for 'mixed' order. More precisely the vector is in digit-reverse order but the digits are internal to the
   * implementation, thus should be considered mixed.
   * This is useful when multiplying polynomials or doing element-wise operations such that the order is agnostic.
   */
  enum class Ordering {
    kNN, /**< Inputs and outputs are in natural-order. */
    kNR, /**< Inputs are in natural-order and outputs are in bit-reversed-order. */
    kRN, /**< Inputs are in bit-reversed-order and outputs are in natural-order. */
    kRR, /**< Inputs and outputs are in bit-reversed-order. */
    kNM, /**< Inputs are in natural-order and outputs are in digit-reversed-order. */
    kMN  /**< Inputs are in digit-reversed-order and outputs are in natural-order. */
  };

  /**
   * @struct NTTConfig
   * @brief Encodes NTT parameters to be passed into the NTT function.
   *
   * @tparam S Type of the coset generator.
   */
  template <typename S>
  struct NTTConfig {
    icicleStreamHandle stream; /**< Stream for asynchronous execution. */
    S coset_gen;               /**< Coset generator. Default value is `S::one()` (no coset). */
    int batch_size;            /**< Number of NTTs to compute. Default value is 1. */
    bool
      columns_batch; /**< True if batches are columns of an input matrix (strided in memory). Default value is false. */
    Ordering ordering;              /**< Ordering of inputs and outputs. Default value is `Ordering::kNN`. */
    bool are_inputs_on_device;      /**< True if inputs are on device, false if on host. Default value is false. */
    bool are_outputs_on_device;     /**< True if outputs are on device, false if on host. Default value is false. */
    bool is_async;                  /**< True if operation is asynchronous. Default value is false. */
    ConfigExtension* ext = nullptr; /**< Backend-specific extensions. */
  };

  /**
   * @brief Returns the default value of NTTConfig for the NTT function.
   *
   * @tparam S Type of the coset generator.
   * @return Default value of NTTConfig.
   */
  template <typename S>
  static NTTConfig<S> default_ntt_config()
  {
    NTTConfig<S> config = {
      nullptr,       // stream
      S::one(),      // coset_gen
      1,             // batch_size
      false,         // columns_batch
      Ordering::kNN, // ordering
      false,         // are_inputs_on_device
      false,         // are_outputs_on_device
      false,         // is_async
    };
    return config;
  }

  /**
   * @struct NTTInitDomainConfig
   * @brief Configuration for initializing the NTT domain.
   */
  struct NTTInitDomainConfig {
    icicleStreamHandle stream;      /**< Stream for asynchronous execution. */
    bool is_async;                  /**< True if operation is asynchronous. Default value is false. */
    ConfigExtension* ext = nullptr; /**< Backend-specific extensions. */
  };

  /**
   * @brief Returns the default value of NTTInitDomainConfig.
   *
   * @return Default value of NTTInitDomainConfig.
   */
  static NTTInitDomainConfig default_ntt_init_domain_config()
  {
    NTTInitDomainConfig config = {
      nullptr, // stream
      false    // is_async
    };
    return config;
  }

  /**
   * @brief Performs the Number Theoretic Transform (NTT).
   *
   * @tparam S Type of the coset generator.
   * @tparam E Type of the elements.
   * @param input Pointer to the input array.
   * @param size Size of the input array.
   * @param dir Direction of the NTT (forward or inverse).
   * @param config Configuration for the NTT operation.
   * @param output Pointer to the output array.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename S, typename E>
  eIcicleError ntt(const E* input, int size, NTTDir dir, NTTConfig<S>& config, E* output);

  /**
   * @brief Initializes the NTT domain.
   *
   * @tparam S Type of the primitive root.
   * @param primitive_root Primitive root of unity.
   * @param config Configuration for initializing the NTT domain.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename S>
  eIcicleError ntt_init_domain(const S& primitive_root, const NTTInitDomainConfig& config);

  /**
   * @brief Releases the NTT domain resources.
   *
   * @tparam S Type of the domain.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename S>
  eIcicleError ntt_release_domain();

  /**
   * @brief Gets the root of unity for a given maximum size.
   *
   * @tparam S Type of the domain.
   * @param max_size Maximum size for the root of unity.
   * @return S Root of unity.
   */
  template <typename S>
  S get_root_of_unity(uint64_t max_size);

  /**
   * @brief Gets the root of unity from the NTT domain for a given logarithmic size.
   *
   * @tparam S Type of the domain.
   * @param logn Logarithmic size.
   * @param rou Pointer to store the root of unity. This is an output parameter.
   * @return eIcicleError Error code indicating success or failure.
   */
  template <typename S>
  eIcicleError get_root_of_unity_from_domain(uint64_t logn, S* rou /*OUT*/);
  /*************************** Backend registration ***************************/

  /*************************** NTT ***************************/
  using NttImpl = std::function<eIcicleError(
    const Device& device, const scalar_t* input, int size, NTTDir dir, NTTConfig<scalar_t>& config, scalar_t* output)>;

  void register_ntt(const std::string& deviceType, NttImpl impl);

#define REGISTER_NTT_BACKEND(DEVICE_TYPE, FUNC)                                                                        \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_ntt) = []() -> bool {                                                                      \
      register_ntt(DEVICE_TYPE, FUNC);                                                                                 \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

#ifdef EXT_FIELD
  using NttExtFieldImpl = std::function<eIcicleError(
    const Device& device,
    const extension_t* input,
    int size,
    NTTDir dir,
    NTTConfig<scalar_t>& config,
    extension_t* output)>;

  void register_extension_ntt(const std::string& deviceType, NttExtFieldImpl impl);

#define REGISTER_NTT_EXT_FIELD_BACKEND(DEVICE_TYPE, FUNC)                                                              \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_ntt_ext_field) = []() -> bool {                                                            \
      register_extension_ntt(DEVICE_TYPE, FUNC);                                                                       \
      return true;                                                                                                     \
    }();                                                                                                               \
  }
#endif // EXT_FIELD

  /*************************** INIT DOMAIN ***************************/
  using NttInitDomainImpl = std::function<eIcicleError(
    const Device& device, const scalar_t& primitive_root, const NTTInitDomainConfig& config)>;

  void register_ntt_init_domain(const std::string& deviceType, NttInitDomainImpl);

#define REGISTER_NTT_INIT_DOMAIN_BACKEND(DEVICE_TYPE, FUNC)                                                            \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_ntt_init_domain) = []() -> bool {                                                          \
      register_ntt_init_domain(DEVICE_TYPE, FUNC);                                                                     \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  /*************************** RELEASE DOMAIN ***************************/
  // Note: 'dummy' is a workaround for the function required per field but need to differentiate by type when
  // calling. TODO Yuval: avoid this param somehow
  using NttReleaseDomainImpl = std::function<eIcicleError(const Device& device, const scalar_t& dummy)>;

  void register_ntt_release_domain(const std::string& deviceType, NttReleaseDomainImpl);

#define REGISTER_NTT_RELEASE_DOMAIN_BACKEND(DEVICE_TYPE, FUNC)                                                         \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_ntt_release_domain) = []() -> bool {                                                       \
      register_ntt_release_domain(DEVICE_TYPE, FUNC);                                                                  \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

  /*************************** GET ROU FROM DOMAIN ***************************/
  using NttGetRouFromDomainImpl = std::function<eIcicleError(const Device& device, uint64_t logn, scalar_t* rou)>;

  void register_ntt_get_rou_from_domain(const std::string& deviceType, NttGetRouFromDomainImpl);

#define REGISTER_NTT_GET_ROU_FROM_DOMAIN_BACKEND(DEVICE_TYPE, FUNC)                                                    \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_ntt_get_rou_from_domain) = []() -> bool {                                                  \
      register_ntt_get_rou_from_domain(DEVICE_TYPE, FUNC);                                                             \
      return true;                                                                                                     \
    }();                                                                                                               \
  }
} // namespace icicle