#pragma once
#include "icicle/errors.h"
#include "icicle/utils/log.h"
#include "icicle/config_extension.h"
#include "icicle/runtime.h"
#include "icicle/fields/field.h"
#include "icicle/utils/utils.h"



#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>

namespace icicle {
    /**
   * @struct FriInitDomainConfig
   * @brief Configuration for initializing the Fri domain.
   */
  struct FriInitDomainConfig {
    icicleStreamHandle stream;      /**< Stream for asynchronous execution. */
    bool is_async;                  /**< True if operation is asynchronous. Default value is false. */
    ConfigExtension* ext = nullptr; /**< Backend-specific extensions. */
  };

  /**
   * @brief Returns the default value of FriInitDomainConfig.
   *
   * @return Default value of FriInitDomainConfig.
   */
  static FriInitDomainConfig default_fri_init_domain_config()
  {
    FriInitDomainConfig config = {
      nullptr, // stream
      false    // is_async
    };
    return config;
  }

  
  template <typename S>
  class FriDomain
  {
    int max_size = 0;
    int max_log_size = 0;
    std::unique_ptr<S[]> twiddles_inv;

  public:
    static eIcicleError fri_init_domain(const S& primitive_root, const FriInitDomainConfig& config);
    static eIcicleError fri_release_domain();
    const inline int get_max_size() const { return max_size; }

    const inline S* get_twiddles_inv() const { return twiddles_inv.get(); }
    static inline FriDomain<S> s_fri_domain;
  };


  /**
   * @brief Initializes the Fri domain.
   *
   * This static function sets up the Fri domain by computing
   * and storing the necessary twiddle factors.
   *
   * @param primitive_root  The primitive root of unity used for Fri computations.
   * @param config          Configuration parameters for initializing the Fri domain, such as
   *                         the maximum log size and other domain-specific settings.
   *
   * @return eIcicleError   Returns `SUCCESS` if the domain is successfully initialized,
   *                         or an appropriate error code if initialization fails.
   */
  template <typename S>
  eIcicleError FriDomain<S>::fri_init_domain(const S& primitive_root, const FriInitDomainConfig& config)
  {
    // (1) check if need to refresh domain. This need to be checked before locking the mutex to avoid unnecessary
    // locking
    if (s_fri_domain.twiddles_inv != nullptr) { return eIcicleError::SUCCESS; }

    // Lock the mutex to ensure thread safety during initialization
    std::lock_guard<std::mutex> lock(s_fri_domain.domain_mutex);

    // Check if domain is already initialized by another thread
    if (s_fri_domain.twiddles_inv == nullptr) {
      // (2) build the domain

      bool found_logn = false;
      S omega = primitive_root;
      const unsigned omegas_count = S::get_omegas_count();
      for (int i = 0; i < omegas_count; i++) {
        omega = S::sqr(omega);
        if (!found_logn) {
          ++s_fri_domain.max_log_size;
          found_logn = omega == S::one();
          if (found_logn) break;
        }
      }

      s_fri_domain.max_size = (int)pow(2, s_fri_domain.max_log_size);
      if (omega != S::one()) {
        ICICLE_LOG_ERROR << "Primitive root provided to the InitDomain function is not a root-of-unity";
        return eIcicleError::INVALID_ARGUMENT;
      }

      // calculate twiddles_inv
      // Note: radix-2 IFri needs ONE in last element (in addition to first element), therefore have n+1 elements

      // Using temp_twiddles_inv to store twiddles_inv before assigning to twiddles_inv using unique_ptr.
      // This is to ensure that twiddles_inv are nullptr during calculation,
      // otherwise the init domain function might return on another thread before twiddles_inv are calculated.
      auto temp_twiddles_inv = std::make_unique<S[]>(s_fri_domain.max_size);
      S primitive_root_inv = S::inv(primitive_root);

      temp_twiddles_inv[0] = S::one();
      for (int i = 1; i < s_fri_domain.max_size; i++) {
        temp_twiddles_inv[i] = temp_twiddles_inv[i - 1] * primitive_root_inv;
      }
      s_fri_domain.twiddles_inv = std::move(temp_twiddles_inv); // Assign twiddles_inv using unique_ptr
    }
    return eIcicleError::SUCCESS;
  }


  /**
   * @brief Releases the resources associated with the NTT domain for the specified device.
   *
   * @param device  The device whose NTT domain resources are to be released.
   *
   * @return eIcicleError  Returns `SUCCESS` if the domain resources are successfully released,
   *                        or an appropriate error code if the release process fails.
   */
  template <typename S>
  eIcicleError FriDomain<S>::fri_release_domain()
  {
    std::lock_guard<std::mutex> lock(s_fri_domain.domain_mutex);
    s_fri_domain.twiddles_inv.reset();            // Set twiddles to nullptr
    s_fri_domain.max_size = 0;
    s_fri_domain.max_log_size = 0;
    return eIcicleError::SUCCESS;
  }


}