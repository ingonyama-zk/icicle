#pragma once
#include "icicle/backend/ntt_backend.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"
#include "icicle/fields/field_config.h"
#include "icicle/vec_ops.h"

#include <thread>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>

using namespace field_config;
using namespace icicle;
namespace ntt_cpu {

  template <typename S>
  class CpuNttDomain
  {
    int max_size = 0;
    int max_log_size = 0;
    std::unique_ptr<S[]> twiddles;
    std::mutex domain_mutex;
    std::unordered_map<S, int> coset_index = {};

  public:
    static eIcicleError
    cpu_ntt_init_domain(const Device& device, const S& primitive_root, const NTTInitDomainConfig& config);
    static eIcicleError cpu_ntt_release_domain(const Device& device);
    static eIcicleError get_root_of_unity_from_domain(const Device& device, uint64_t logn, S* rou /*OUT*/);

    // template <typename U, typename E>
    // eIcicleError
    // cpu_ntt_ref(const Device& device, const E* input, uint64_t size, NTTDir dir, NTTConfig<S>& config, E* output);

    template <typename U, typename E>
    eIcicleError
    cpu_ntt(const Device& device, const E* input, uint64_t size, NTTDir dir, NTTConfig<S>& config, E* output);

    const S* get_twiddles() const { return twiddles.get(); }
    const int get_max_size() const { return max_size; }
    const uint64_t get_coset_stride(const S& key) const { return coset_index.at(key); }
    static inline CpuNttDomain<S> s_ntt_domain;
  };

  template <typename S>
  eIcicleError
  CpuNttDomain<S>::cpu_ntt_init_domain(const Device& device, const S& primitive_root, const NTTInitDomainConfig& config)
  {
    // (1) check if need to refresh domain. This need to be checked before locking the mutex to avoid unnecessary
    // locking
    if (s_ntt_domain.twiddles != nullptr) { return eIcicleError::SUCCESS; }

    // Lock the mutex to ensure thread safety during initialization
    std::lock_guard<std::mutex> lock(s_ntt_domain.domain_mutex);

    // Check if domain is already initialized by another thread
    if (s_ntt_domain.twiddles == nullptr) {
      // (2) build the domain

      // bool found_logn = false;
      // S omega = primitive_root;
      // const unsigned omegas_count = S::get_omegas_count();
      // for (int i = 0; i < omegas_count; i++) {
      //   omega = S::sqr(omega);
      //   if (!found_logn) {
      //     ++s_ntt_domain.max_log_size;
      //     found_logn = omega == S::one();
      //     if (found_logn) break;
      //   }
      // }
      s_ntt_domain.max_log_size = 21;

      s_ntt_domain.max_size = (int)pow(2, s_ntt_domain.max_log_size);
      // if (omega != S::one()) {
      //   ICICLE_LOG_ERROR << "Primitive root provided to the InitDomain function is not a root-of-unity";
      //   return eIcicleError::INVALID_ARGUMENT;
      // }

      // calculate twiddles
      // Note: radix-2 INTT needs ONE in last element (in addition to first element), therefore have n+1 elements

      // Using temp_twiddles to store twiddles before assigning to twiddles using unique_ptr.
      // This is to ensure that twiddles are nullptr during calculation,
      // otherwise the init domain function might return on another thread before twiddles are calculated.
      auto temp_twiddles = std::make_unique<S[]>(s_ntt_domain.max_size + 1);

      S tw_omega = primitive_root;
      temp_twiddles[0] = S::one();
      for (int i = 1; i <= s_ntt_domain.max_size; i++) {
        temp_twiddles[i] = temp_twiddles[i - 1] * tw_omega;
        s_ntt_domain.coset_index[temp_twiddles[i]] = i;
      }
      s_ntt_domain.twiddles = std::move(temp_twiddles); // Assign twiddles using unique_ptr
    }
    return eIcicleError::SUCCESS;
  }

  template <typename S>
  eIcicleError CpuNttDomain<S>::cpu_ntt_release_domain(const Device& device)
  {
    std::lock_guard<std::mutex> lock(s_ntt_domain.domain_mutex);
    s_ntt_domain.twiddles.reset(); // Set twiddles to nullptr
    s_ntt_domain.max_size = 0;
    s_ntt_domain.max_log_size = 0;
    s_ntt_domain.coset_index.clear();
    return eIcicleError::SUCCESS;
  }

  template <typename S>
  eIcicleError CpuNttDomain<S>::get_root_of_unity_from_domain(const Device& device, uint64_t logn, S* rou /*OUT*/)
  {
    std::lock_guard<std::mutex> lock(s_ntt_domain.domain_mutex); // not ideal to lock here but safer
    ICICLE_ASSERT(logn <= s_ntt_domain.max_log_size)
      << "NTT log_size=" << logn << " is too large for the domain (logsize=" << s_ntt_domain.max_log_size
      << "). Consider generating your domain with a higher order root of unity";

    const size_t twiddles_idx = 1ULL << (s_ntt_domain.max_log_size - logn);
    *rou = s_ntt_domain.twiddles[twiddles_idx];
    return eIcicleError::SUCCESS;
  }

} // namespace ntt_cpu