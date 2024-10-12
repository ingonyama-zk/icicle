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
    std::unique_ptr<S[]> winograd8_twiddles;
    std::unique_ptr<S[]> winograd8_twiddles_inv;
    std::unique_ptr<S[]> winograd16_twiddles;
    std::unique_ptr<S[]> winograd16_twiddles_inv;
    std::unique_ptr<S[]> winograd32_twiddles;
    std::unique_ptr<S[]> winograd32_twiddles_inv;
    std::mutex domain_mutex;
    std::unordered_map<S, int> coset_index = {};

  public:
    static eIcicleError
    cpu_ntt_init_domain(const Device& device, const S& primitive_root, const NTTInitDomainConfig& config);
    static eIcicleError cpu_ntt_release_domain(const Device& device);
    static eIcicleError get_root_of_unity_from_domain(const Device& device, uint64_t logn, S* rou /*OUT*/);

    const inline S* get_twiddles() const { return twiddles.get(); }
    const inline S* get_winograd8_twiddles() const { return winograd8_twiddles.get(); }
    const inline S* get_winograd8_twiddles_inv() const { return winograd8_twiddles_inv.get(); }
    const inline S* get_winograd16_twiddles() const { return winograd16_twiddles.get(); }
    const inline S* get_winograd16_twiddles_inv() const { return winograd16_twiddles_inv.get(); }
    const inline S* get_winograd32_twiddles() const { return winograd32_twiddles.get(); }
    const inline S* get_winograd32_twiddles_inv() const { return winograd32_twiddles_inv.get(); }
    const inline int get_max_size() const { return max_size; }
    const inline uint64_t get_coset_stride(const S& key) const { return coset_index.at(key); }
    static inline CpuNttDomain<S> s_ntt_domain;
  };

  /**
   * @brief Initializes the NTT domain with the specified device, primitive root, and configuration.
   *
   * This static function sets up the NTT domain by computing
   * and storing the necessary twiddle factors and other precomputed values required for
   * performing NTT operations.
   *
   * @param device          The device on which the NTT domain is being initialized.
   * @param primitive_root  The primitive root of unity used for NTT computations.
   * @param config          Configuration parameters for initializing the NTT domain, such as
   *                         the maximum log size and other domain-specific settings.
   *
   * @return eIcicleError   Returns `SUCCESS` if the domain is successfully initialized,
   *                         or an appropriate error code if initialization fails.
   */
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

      bool found_logn = false;
      S omega = primitive_root;
      const unsigned omegas_count = S::get_omegas_count();
      for (int i = 0; i < omegas_count; i++) {
        omega = S::sqr(omega);
        if (!found_logn) {
          ++s_ntt_domain.max_log_size;
          found_logn = omega == S::one();
          if (found_logn) break;
        }
      }

      s_ntt_domain.max_size = (int)pow(2, s_ntt_domain.max_log_size);
      if (omega != S::one()) {
        ICICLE_LOG_ERROR << "Primitive root provided to the InitDomain function is not a root-of-unity";
        return eIcicleError::INVALID_ARGUMENT;
      }

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

      // Winograd 8
      if (s_ntt_domain.max_log_size >= 3) {
        auto temp_win8_twiddles = std::make_unique<S[]>(3);
        auto temp_win8_twiddles_inv = std::make_unique<S[]>(3);
        int basic_tw_idx = (s_ntt_domain.max_size >> 3);
        S basic_tw = s_ntt_domain.twiddles[basic_tw_idx];
        temp_win8_twiddles[0] = basic_tw * basic_tw;
        temp_win8_twiddles[1] = (basic_tw + temp_win8_twiddles[0] * basic_tw) * S::inv_log_size(1);
        temp_win8_twiddles[2] =
          (basic_tw - temp_win8_twiddles[0] * basic_tw) * S::inv_log_size(1);   // = temp_win8_twiddles_inv[2]
        basic_tw = s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx]; // for inverse ntt
        temp_win8_twiddles_inv[0] = basic_tw * basic_tw;                        // temp_win8_twiddles_inv[0]
        temp_win8_twiddles_inv[1] =
          (basic_tw + temp_win8_twiddles_inv[0] * basic_tw) * S::inv_log_size(1); // temp_win8_twiddles_inv[1]
        temp_win8_twiddles_inv[2] = temp_win8_twiddles[2];

        s_ntt_domain.winograd8_twiddles = std::move(temp_win8_twiddles);         // Assign twiddles using unique_ptr
        s_ntt_domain.winograd8_twiddles_inv = std::move(temp_win8_twiddles_inv); // Assign twiddles using unique_ptr
      }

      // Winograd 16
      if (s_ntt_domain.max_log_size >= 4) {
        auto temp_win16_twiddles = std::make_unique<S[]>(18);
        auto temp_win16_twiddles_inv = std::make_unique<S[]>(18);
        int basic_tw_idx = s_ntt_domain.max_size >> 4;

        temp_win16_twiddles[0] = s_ntt_domain.twiddles[basic_tw_idx * 0];
        temp_win16_twiddles[1] = s_ntt_domain.twiddles[basic_tw_idx * 0];
        temp_win16_twiddles[2] = s_ntt_domain.twiddles[basic_tw_idx * 0];
        temp_win16_twiddles[3] = s_ntt_domain.twiddles[basic_tw_idx * 4];

        temp_win16_twiddles[4] = s_ntt_domain.twiddles[basic_tw_idx * 0];
        temp_win16_twiddles[5] = s_ntt_domain.twiddles[basic_tw_idx * 4];
        temp_win16_twiddles[6] =
          (s_ntt_domain.twiddles[basic_tw_idx * 2] + s_ntt_domain.twiddles[basic_tw_idx * 6]) * S::inv_log_size(1);
        temp_win16_twiddles[7] =
          (s_ntt_domain.twiddles[basic_tw_idx * 2] - s_ntt_domain.twiddles[basic_tw_idx * 6]) * S::inv_log_size(1);

        temp_win16_twiddles[8] = s_ntt_domain.twiddles[basic_tw_idx * 0];
        temp_win16_twiddles[9] = s_ntt_domain.twiddles[basic_tw_idx * 4];
        temp_win16_twiddles[10] =
          (s_ntt_domain.twiddles[basic_tw_idx * 2] + s_ntt_domain.twiddles[basic_tw_idx * 6]) * S::inv_log_size(1);
        temp_win16_twiddles[11] =
          (s_ntt_domain.twiddles[basic_tw_idx * 2] - s_ntt_domain.twiddles[basic_tw_idx * 6]) * S::inv_log_size(1);

        temp_win16_twiddles[12] = (s_ntt_domain.twiddles[basic_tw_idx * 1] - s_ntt_domain.twiddles[basic_tw_idx * 3] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 5] + s_ntt_domain.twiddles[basic_tw_idx * 7]) *
                                  S::inv_log_size(1);
        temp_win16_twiddles[13] = S::neg(
                                    s_ntt_domain.twiddles[basic_tw_idx * 1] + s_ntt_domain.twiddles[basic_tw_idx * 3] +
                                    s_ntt_domain.twiddles[basic_tw_idx * 5] + s_ntt_domain.twiddles[basic_tw_idx * 7]) *
                                  S::inv_log_size(1);
        temp_win16_twiddles[14] =
          (s_ntt_domain.twiddles[basic_tw_idx * 3] + s_ntt_domain.twiddles[basic_tw_idx * 5]) * S::inv_log_size(1);

        temp_win16_twiddles[15] = (s_ntt_domain.twiddles[basic_tw_idx * 1] - s_ntt_domain.twiddles[basic_tw_idx * 3] +
                                   s_ntt_domain.twiddles[basic_tw_idx * 5] - s_ntt_domain.twiddles[basic_tw_idx * 7]) *
                                  S::inv_log_size(1);
        temp_win16_twiddles[16] = (s_ntt_domain.twiddles[basic_tw_idx * 5] - s_ntt_domain.twiddles[basic_tw_idx * 1] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 3] + s_ntt_domain.twiddles[basic_tw_idx * 7]) *
                                  S::inv_log_size(1);
        temp_win16_twiddles[17] =
          (s_ntt_domain.twiddles[basic_tw_idx * 3] - s_ntt_domain.twiddles[basic_tw_idx * 5]) * S::inv_log_size(1);

        temp_win16_twiddles_inv[0] = s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 0];
        temp_win16_twiddles_inv[1] = s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 0];
        temp_win16_twiddles_inv[2] = s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 0];
        temp_win16_twiddles_inv[3] = s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 4];

        temp_win16_twiddles_inv[4] = s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 0];
        temp_win16_twiddles_inv[5] = s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 4];
        temp_win16_twiddles_inv[6] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 2] +
                                      s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 6]) *
                                     S::inv_log_size(1);
        temp_win16_twiddles_inv[7] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 2] -
                                      s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 6]) *
                                     S::inv_log_size(1);

        temp_win16_twiddles_inv[8] = s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 0];
        temp_win16_twiddles_inv[9] = s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 4];
        temp_win16_twiddles_inv[10] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 2] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 6]) *
                                      S::inv_log_size(1);
        temp_win16_twiddles_inv[11] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 2] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 6]) *
                                      S::inv_log_size(1);

        temp_win16_twiddles_inv[12] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 1] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 7]) *
                                      S::inv_log_size(1);
        temp_win16_twiddles_inv[13] = S::neg(
                                        s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 1] +
                                        s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] +
                                        s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5] +
                                        s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 7]) *
                                      S::inv_log_size(1);
        temp_win16_twiddles_inv[14] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5]) *
                                      S::inv_log_size(1);

        temp_win16_twiddles_inv[15] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 1] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 7]) *
                                      S::inv_log_size(1);
        temp_win16_twiddles_inv[16] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 1] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 7]) *
                                      S::inv_log_size(1);
        temp_win16_twiddles_inv[17] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5]) *
                                      S::inv_log_size(1);

        s_ntt_domain.winograd16_twiddles = std::move(temp_win16_twiddles);         // Assign twiddles using unique_ptr
        s_ntt_domain.winograd16_twiddles_inv = std::move(temp_win16_twiddles_inv); // Assign twiddles using unique_ptr
      }

      // Winograd 32
      if (s_ntt_domain.max_log_size >= 5) {
        auto temp_win32_twiddles = std::make_unique<S[]>(46);
        auto temp_win32_twiddles_inv = std::make_unique<S[]>(46);
        int basic_tw_idx = s_ntt_domain.max_size >> 4;
        temp_win32_twiddles[0] = s_ntt_domain.twiddles[basic_tw_idx * 0];
        temp_win32_twiddles[1] = s_ntt_domain.twiddles[basic_tw_idx * 0];
        temp_win32_twiddles[2] = s_ntt_domain.twiddles[basic_tw_idx * 0];
        temp_win32_twiddles[3] = s_ntt_domain.twiddles[basic_tw_idx * 4];

        temp_win32_twiddles[4] = s_ntt_domain.twiddles[basic_tw_idx * 0];
        temp_win32_twiddles[5] = s_ntt_domain.twiddles[basic_tw_idx * 4];
        temp_win32_twiddles[6] =
          (s_ntt_domain.twiddles[basic_tw_idx * 2] + s_ntt_domain.twiddles[basic_tw_idx * 6]) * S::inv_log_size(1);
        temp_win32_twiddles[7] =
          (s_ntt_domain.twiddles[basic_tw_idx * 2] - s_ntt_domain.twiddles[basic_tw_idx * 6]) * S::inv_log_size(1);

        temp_win32_twiddles[8] = s_ntt_domain.twiddles[basic_tw_idx * 0];
        temp_win32_twiddles[9] = s_ntt_domain.twiddles[basic_tw_idx * 4];
        temp_win32_twiddles[10] =
          (s_ntt_domain.twiddles[basic_tw_idx * 2] + s_ntt_domain.twiddles[basic_tw_idx * 6]) * S::inv_log_size(1);
        temp_win32_twiddles[11] =
          (s_ntt_domain.twiddles[basic_tw_idx * 2] - s_ntt_domain.twiddles[basic_tw_idx * 6]) * S::inv_log_size(1);

        temp_win32_twiddles[12] = (s_ntt_domain.twiddles[basic_tw_idx * 1] - s_ntt_domain.twiddles[basic_tw_idx * 3] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 5] + s_ntt_domain.twiddles[basic_tw_idx * 7]) *
                                  S::inv_log_size(1);
        temp_win32_twiddles[13] = S::neg(
                                    s_ntt_domain.twiddles[basic_tw_idx * 1] + s_ntt_domain.twiddles[basic_tw_idx * 3] +
                                    s_ntt_domain.twiddles[basic_tw_idx * 5] + s_ntt_domain.twiddles[basic_tw_idx * 7]) *
                                  S::inv_log_size(1);
        temp_win32_twiddles[14] =
          (s_ntt_domain.twiddles[basic_tw_idx * 3] + s_ntt_domain.twiddles[basic_tw_idx * 5]) * S::inv_log_size(1);

        temp_win32_twiddles[15] = (s_ntt_domain.twiddles[basic_tw_idx * 1] - s_ntt_domain.twiddles[basic_tw_idx * 3] +
                                   s_ntt_domain.twiddles[basic_tw_idx * 5] - s_ntt_domain.twiddles[basic_tw_idx * 7]) *
                                  S::inv_log_size(1);
        temp_win32_twiddles[16] = (s_ntt_domain.twiddles[basic_tw_idx * 5] - s_ntt_domain.twiddles[basic_tw_idx * 1] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 3] + s_ntt_domain.twiddles[basic_tw_idx * 7]) *
                                  S::inv_log_size(1);
        temp_win32_twiddles[17] =
          (s_ntt_domain.twiddles[basic_tw_idx * 3] - s_ntt_domain.twiddles[basic_tw_idx * 5]) * S::inv_log_size(1);

        basic_tw_idx = s_ntt_domain.max_size >> 5;

        temp_win32_twiddles[18] = s_ntt_domain.twiddles[basic_tw_idx * 0];
        temp_win32_twiddles[19] = s_ntt_domain.twiddles[basic_tw_idx * 8];

        temp_win32_twiddles[20] =
          (s_ntt_domain.twiddles[basic_tw_idx * 4] + s_ntt_domain.twiddles[basic_tw_idx * 12]) * S::inv_log_size(1);
        temp_win32_twiddles[21] =
          (s_ntt_domain.twiddles[basic_tw_idx * 4] - s_ntt_domain.twiddles[basic_tw_idx * 12]) * S::inv_log_size(1);

        temp_win32_twiddles[22] = (s_ntt_domain.twiddles[basic_tw_idx * 2] + s_ntt_domain.twiddles[basic_tw_idx * 14] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 6] - s_ntt_domain.twiddles[basic_tw_idx * 10]) *
                                  S::inv_log_size(1);
        temp_win32_twiddles[23] =
          S::neg(
            s_ntt_domain.twiddles[basic_tw_idx * 2] + s_ntt_domain.twiddles[basic_tw_idx * 14] +
            s_ntt_domain.twiddles[basic_tw_idx * 6] + s_ntt_domain.twiddles[basic_tw_idx * 10]) *
          S::inv_log_size(1);

        temp_win32_twiddles[24] =
          (s_ntt_domain.twiddles[basic_tw_idx * 6] + s_ntt_domain.twiddles[basic_tw_idx * 10]) * S::inv_log_size(1);

        temp_win32_twiddles[25] = (s_ntt_domain.twiddles[basic_tw_idx * 2] - s_ntt_domain.twiddles[basic_tw_idx * 14] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 6] + s_ntt_domain.twiddles[basic_tw_idx * 10]) *
                                  S::inv_log_size(1);
        temp_win32_twiddles[26] = (s_ntt_domain.twiddles[basic_tw_idx * 14] - s_ntt_domain.twiddles[basic_tw_idx * 2] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 6] + s_ntt_domain.twiddles[basic_tw_idx * 10]) *
                                  S::inv_log_size(1);

        temp_win32_twiddles[27] =
          (s_ntt_domain.twiddles[basic_tw_idx * 6] - s_ntt_domain.twiddles[basic_tw_idx * 10]) * S::inv_log_size(1);

        temp_win32_twiddles[28] = (s_ntt_domain.twiddles[basic_tw_idx * 1] + s_ntt_domain.twiddles[basic_tw_idx * 15] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 5] - s_ntt_domain.twiddles[basic_tw_idx * 11] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 7] - s_ntt_domain.twiddles[basic_tw_idx * 9] +
                                   s_ntt_domain.twiddles[basic_tw_idx * 3] + s_ntt_domain.twiddles[basic_tw_idx * 13]) *
                                  S::inv_log_size(1);
        temp_win32_twiddles[29] = (s_ntt_domain.twiddles[basic_tw_idx * 5] - s_ntt_domain.twiddles[basic_tw_idx * 1] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 15] + s_ntt_domain.twiddles[basic_tw_idx * 11] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 7] - s_ntt_domain.twiddles[basic_tw_idx * 9] +
                                   s_ntt_domain.twiddles[basic_tw_idx * 3] + s_ntt_domain.twiddles[basic_tw_idx * 13]) *
                                  S::inv_log_size(1);
        temp_win32_twiddles[30] = (s_ntt_domain.twiddles[basic_tw_idx * 7] + s_ntt_domain.twiddles[basic_tw_idx * 9] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 3] - s_ntt_domain.twiddles[basic_tw_idx * 13]) *
                                  S::inv_log_size(1);
        temp_win32_twiddles[31] =
          (s_ntt_domain.twiddles[basic_tw_idx * 3] - s_ntt_domain.twiddles[basic_tw_idx * 7] -
           s_ntt_domain.twiddles[basic_tw_idx * 9] - s_ntt_domain.twiddles[basic_tw_idx * 5] -
           s_ntt_domain.twiddles[basic_tw_idx * 11] - s_ntt_domain.twiddles[basic_tw_idx * 1] -
           s_ntt_domain.twiddles[basic_tw_idx * 15] + s_ntt_domain.twiddles[basic_tw_idx * 13]) *
          S::inv_log_size(1);
        temp_win32_twiddles[32] = (s_ntt_domain.twiddles[basic_tw_idx * 7] + s_ntt_domain.twiddles[basic_tw_idx * 9] +
                                   s_ntt_domain.twiddles[basic_tw_idx * 5] + s_ntt_domain.twiddles[basic_tw_idx * 11] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 1] - s_ntt_domain.twiddles[basic_tw_idx * 15] +
                                   s_ntt_domain.twiddles[basic_tw_idx * 3] + s_ntt_domain.twiddles[basic_tw_idx * 13]) *
                                  S::inv_log_size(1);
        temp_win32_twiddles[33] = (s_ntt_domain.twiddles[basic_tw_idx * 1] + s_ntt_domain.twiddles[basic_tw_idx * 15] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 3] - s_ntt_domain.twiddles[basic_tw_idx * 13]) *
                                  S::inv_log_size(1);
        temp_win32_twiddles[34] = (s_ntt_domain.twiddles[basic_tw_idx * 5] + s_ntt_domain.twiddles[basic_tw_idx * 11] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 3] - s_ntt_domain.twiddles[basic_tw_idx * 13]) *
                                  S::inv_log_size(1);
        temp_win32_twiddles[35] =
          (S::neg(
            s_ntt_domain.twiddles[basic_tw_idx * 5] + s_ntt_domain.twiddles[basic_tw_idx * 11] +
            s_ntt_domain.twiddles[basic_tw_idx * 3] + s_ntt_domain.twiddles[basic_tw_idx * 13])) *
          S::inv_log_size(1);
        temp_win32_twiddles[36] =
          (s_ntt_domain.twiddles[basic_tw_idx * 3] + s_ntt_domain.twiddles[basic_tw_idx * 13]) * S::inv_log_size(1);
        temp_win32_twiddles[37] = (s_ntt_domain.twiddles[basic_tw_idx * 1] - s_ntt_domain.twiddles[basic_tw_idx * 15] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 5] + s_ntt_domain.twiddles[basic_tw_idx * 11] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 7] + s_ntt_domain.twiddles[basic_tw_idx * 9] +
                                   s_ntt_domain.twiddles[basic_tw_idx * 3] - s_ntt_domain.twiddles[basic_tw_idx * 13]) *
                                  S::inv_log_size(1);

        temp_win32_twiddles[38] = (s_ntt_domain.twiddles[basic_tw_idx * 15] - s_ntt_domain.twiddles[basic_tw_idx * 1] +
                                   s_ntt_domain.twiddles[basic_tw_idx * 5] - s_ntt_domain.twiddles[basic_tw_idx * 11] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 7] + s_ntt_domain.twiddles[basic_tw_idx * 9] +
                                   s_ntt_domain.twiddles[basic_tw_idx * 3] - s_ntt_domain.twiddles[basic_tw_idx * 13]) *
                                  S::inv_log_size(1);

        temp_win32_twiddles[39] = (s_ntt_domain.twiddles[basic_tw_idx * 7] - s_ntt_domain.twiddles[basic_tw_idx * 9] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 3] + s_ntt_domain.twiddles[basic_tw_idx * 13]) *
                                  S::inv_log_size(1);
        temp_win32_twiddles[40] = (s_ntt_domain.twiddles[basic_tw_idx * 7] - s_ntt_domain.twiddles[basic_tw_idx * 9] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 5] + s_ntt_domain.twiddles[basic_tw_idx * 11] +
                                   s_ntt_domain.twiddles[basic_tw_idx * 1] - s_ntt_domain.twiddles[basic_tw_idx * 15] +
                                   s_ntt_domain.twiddles[basic_tw_idx * 3] - s_ntt_domain.twiddles[basic_tw_idx * 13]) *
                                  S::inv_log_size(1);
        temp_win32_twiddles[41] = (s_ntt_domain.twiddles[basic_tw_idx * 9] - s_ntt_domain.twiddles[basic_tw_idx * 7] +
                                   s_ntt_domain.twiddles[basic_tw_idx * 5] - s_ntt_domain.twiddles[basic_tw_idx * 11] +
                                   s_ntt_domain.twiddles[basic_tw_idx * 1] - s_ntt_domain.twiddles[basic_tw_idx * 15] +
                                   s_ntt_domain.twiddles[basic_tw_idx * 3] - s_ntt_domain.twiddles[basic_tw_idx * 13]) *
                                  S::inv_log_size(1);
        temp_win32_twiddles[42] = (s_ntt_domain.twiddles[basic_tw_idx * 15] - s_ntt_domain.twiddles[basic_tw_idx * 1] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 3] + s_ntt_domain.twiddles[basic_tw_idx * 13]) *
                                  S::inv_log_size(1);
        temp_win32_twiddles[43] = (s_ntt_domain.twiddles[basic_tw_idx * 5] - s_ntt_domain.twiddles[basic_tw_idx * 11] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 3] + s_ntt_domain.twiddles[basic_tw_idx * 13]) *
                                  S::inv_log_size(1);
        temp_win32_twiddles[44] = (s_ntt_domain.twiddles[basic_tw_idx * 11] - s_ntt_domain.twiddles[basic_tw_idx * 5] -
                                   s_ntt_domain.twiddles[basic_tw_idx * 3] + s_ntt_domain.twiddles[basic_tw_idx * 13]) *
                                  S::inv_log_size(1);
        temp_win32_twiddles[45] =
          (s_ntt_domain.twiddles[basic_tw_idx * 3] - s_ntt_domain.twiddles[basic_tw_idx * 13]) * S::inv_log_size(1);

        s_ntt_domain.winograd32_twiddles = std::move(temp_win32_twiddles); // Assign twiddles using unique_ptr

        basic_tw_idx = s_ntt_domain.max_size >> 4;
        temp_win32_twiddles_inv[0] = s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 0];
        temp_win32_twiddles_inv[1] = s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 0];
        temp_win32_twiddles_inv[2] = s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 0];
        temp_win32_twiddles_inv[3] = s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 4];

        temp_win32_twiddles_inv[4] = s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 0];
        temp_win32_twiddles_inv[5] = s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 4];
        temp_win32_twiddles_inv[6] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 2] +
                                      s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 6]) *
                                     S::inv_log_size(1);
        temp_win32_twiddles_inv[7] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 2] -
                                      s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 6]) *
                                     S::inv_log_size(1);

        temp_win32_twiddles_inv[8] = s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 0];
        temp_win32_twiddles_inv[9] = s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 4];
        temp_win32_twiddles_inv[10] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 2] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 6]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[11] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 2] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 6]) *
                                      S::inv_log_size(1);

        temp_win32_twiddles_inv[12] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 1] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 7]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[13] = S::neg(
                                        s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 1] +
                                        s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] +
                                        s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5] +
                                        s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 7]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[14] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5]) *
                                      S::inv_log_size(1);

        temp_win32_twiddles_inv[15] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 1] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 7]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[16] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 1] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 7]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[17] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5]) *
                                      S::inv_log_size(1);

        basic_tw_idx = s_ntt_domain.max_size >> 5;

        temp_win32_twiddles_inv[18] = s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 0];
        temp_win32_twiddles_inv[19] = s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 8];

        temp_win32_twiddles_inv[20] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 4] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 12]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[21] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 4] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 12]) *
                                      S::inv_log_size(1);

        temp_win32_twiddles_inv[22] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 2] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 14] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 6] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 10]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[23] = S::neg(
                                        s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 2] +
                                        s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 14] +
                                        s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 6] +
                                        s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 10]) *
                                      S::inv_log_size(1);

        temp_win32_twiddles_inv[24] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 6] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 10]) *
                                      S::inv_log_size(1);

        temp_win32_twiddles_inv[25] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 2] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 14] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 6] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 10]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[26] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 14] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 2] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 6] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 10]) *
                                      S::inv_log_size(1);

        temp_win32_twiddles_inv[27] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 6] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 10]) *
                                      S::inv_log_size(1);

        temp_win32_twiddles_inv[28] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 1] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 15] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 11] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 7] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 9] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 13]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[29] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 1] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 15] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 11] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 7] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 9] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 13]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[30] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 7] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 9] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 13]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[31] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 7] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 9] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 11] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 1] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 15] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 13]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[32] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 7] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 9] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 11] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 1] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 15] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 13]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[33] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 1] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 15] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 13]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[34] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 11] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 13]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[35] = (S::neg(
                                        s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5] +
                                        s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 11] +
                                        s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] +
                                        s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 13])) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[36] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 13]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[37] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 1] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 15] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 11] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 7] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 9] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 13]) *
                                      S::inv_log_size(1);

        temp_win32_twiddles_inv[38] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 15] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 1] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 11] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 7] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 9] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 13]) *
                                      S::inv_log_size(1);

        temp_win32_twiddles_inv[39] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 7] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 9] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 13]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[40] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 7] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 9] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 11] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 1] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 15] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 13]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[41] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 9] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 7] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 11] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 1] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 15] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 13]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[42] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 15] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 1] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 13]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[43] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 11] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 13]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[44] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 11] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 5] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] +
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 13]) *
                                      S::inv_log_size(1);
        temp_win32_twiddles_inv[45] = (s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 3] -
                                       s_ntt_domain.twiddles[s_ntt_domain.max_size - basic_tw_idx * 13]) *
                                      S::inv_log_size(1);

        s_ntt_domain.winograd32_twiddles_inv = std::move(temp_win32_twiddles_inv); // Assign twiddles using unique_ptr
      }
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
  eIcicleError CpuNttDomain<S>::cpu_ntt_release_domain(const Device& device)
  {
    std::lock_guard<std::mutex> lock(s_ntt_domain.domain_mutex);
    s_ntt_domain.twiddles.reset();                // Set twiddles to nullptr
    s_ntt_domain.winograd8_twiddles.reset();      // Set winograd8_twiddles to nullptr
    s_ntt_domain.winograd16_twiddles.reset();     // Set winograd16_twiddles to nullptr
    s_ntt_domain.winograd32_twiddles.reset();     // Set winograd32_twiddles to nullptr
    s_ntt_domain.winograd8_twiddles_inv.reset();  // Set winograd8_twiddles to nullptr
    s_ntt_domain.winograd16_twiddles_inv.reset(); // Set winograd16_twiddles to nullptr
    s_ntt_domain.winograd32_twiddles_inv.reset(); // Set winograd32_twiddles_inv to nullptr
    s_ntt_domain.max_size = 0;
    s_ntt_domain.max_log_size = 0;
    s_ntt_domain.coset_index.clear();
    return eIcicleError::SUCCESS;
  }

  /**
   * @brief Retrieves the root of unity for a given log size from the initialized NTT domain.
   *
   * @param device  The device associated with the NTT domain.
   * @param logn    The log of the size for which the root of unity is requested.
   * @param rou     Pointer to a variable where the retrieved root of unity will be stored.
   *
   * @return eIcicleError  Returns `SUCCESS` if the root of unity is successfully retrieved,
   *                        or an appropriate error code if the retrieval fails (e.g., if `logn`
   *                        exceeds the maximum log size).
   *
   * @note The caller must ensure that the NTT domain has been initialized before calling this function.
   */
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