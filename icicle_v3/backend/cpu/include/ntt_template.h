#include "icicle/ntt.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"

#include "icicle/fields/field_config.h"
#include "icicle/utils/log.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstdint>

using namespace field_config;
using namespace icicle;
namespace ntt_template {
  template <typename S>
  class CpuNttDomain {
      //TODO - coset, mixed-radix NTT
      // Mutex for protecting access to the domain/device container array
      // static inline std::mutex cpu_ntt_domain_mutex; //TODO
      // The domain-per-device container - assumption is init_domain is called once per device per program.

      int max_size = 0;
      int max_log_size = 0;
      S* twiddles = nullptr;
      
      public:
          template <typename U>
          friend eIcicleError cpu_ntt_init_domain(const Device& device, const U& primitive_root, const NTTInitDomainConfig& config);
          
          template <typename U>
          friend eIcicleError generate_twiddles(const U& primitive_root, U* twiddles, int logn);

          template <typename U>
          friend eIcicleError cpu_ntt_release_domain(const Device& device);

          template <typename U, typename E>
          friend eIcicleError cpu_ntt_ref(const Device& device, const E* input, int size, NTTDir dir, NTTConfig<S>& config, E* output);
          
          template <typename U, typename E>
          friend eIcicleError cpu_ntt(const Device& device, const E* input, int size, NTTDir dir, NTTConfig<S>& config, E* output);

          S* get_twiddles() {
              return twiddles;
          }

  };

  template <typename S>
  static CpuNttDomain<S>* s_ntt_domain = nullptr;

  template <typename S = scalar_t>
  eIcicleError generate_twiddles(const S& primitive_root, S* twiddles, int n)
  {
      S omega = primitive_root;
      twiddles[0] = S::one();
      for (int i = 1; i <= n; i++) {
          twiddles[i] = twiddles[i - 1] * omega;
      }
      return eIcicleError::SUCCESS;
  }

  template <typename S = scalar_t>
  eIcicleError cpu_ntt_init_domain(const Device& device, const S& primitive_root, const NTTInitDomainConfig& config)
  {
      // (1) check if need to refresh domain. If no, return, else build new domain
      if (s_ntt_domain<S> != nullptr) {
          return eIcicleError::SUCCESS;
      }
      // (2) build the domain
      delete s_ntt_domain<S>;
      s_ntt_domain<S> = new CpuNttDomain<S>();
      
      bool found_logn = false;
      S omega = primitive_root;
      unsigned omegas_count = S::get_omegas_count();
      for (int i = 0; i < omegas_count; i++) {
          omega = S::sqr(omega);
          if (!found_logn) {
              ++s_ntt_domain<S>->max_log_size;
              found_logn = omega == S::one();
              if (found_logn) break;
          }
      }

      s_ntt_domain<S>->max_size = (int)pow(2, s_ntt_domain<S>->max_log_size);
      if (omega != S::one()) {
          ICICLE_LOG_ERROR << "Primitive root provided to the InitDomain function is not in the subgroup";
          return eIcicleError::INVALID_ARGUMENT;
      }

      // calculate twiddles
      // Note: radix-2 INTT needs ONE in last element (in addition to first element), therefore have n+1 elements

      s_ntt_domain<S>->twiddles = (S*)malloc((s_ntt_domain<S>->max_size + 1) * sizeof(S));
      generate_twiddles(primitive_root, s_ntt_domain<S>->twiddles, s_ntt_domain<S>->max_size);
      return eIcicleError::SUCCESS;
  }

  template <typename S = scalar_t>
  eIcicleError cpu_ntt_release_domain(const Device& device)
  {
      // release the memory
      free(s_ntt_domain<S>->twiddles);
      delete s_ntt_domain<S>; 
      s_ntt_domain<S> = nullptr;
      return eIcicleError::SUCCESS;
  }

  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError cpu_ntt_ref(const Device& device, const E* input, int size, NTTDir dir, NTTConfig<S>& config, E* output)
  {
      ICICLE_LOG_DEBUG << "CPU NTT REF";
      size = size/config.batch_size;
      if (size & (size - 1)) {
          return eIcicleError::INVALID_ARGUMENT; 
      }

      std::copy(input, input + size, output);

      ICICLE_LOG_DEBUG << "INITIAL OUTPUT (REF)";
      for (int i = 0; i < size; ++i) {
          ICICLE_LOG_DEBUG << "output[" << i << "]: " << output[i];
      }

      // Bit-reversal permutation
      int logn = 0;
      for (int n = size; n > 1; n >>= 1) {
          logn++;
      }

      for (int i = 0; i < size; ++i) {
          int rev = 0;
          for (int j = 0; j < logn; ++j) {
              if (i & (1 << j)) {
                  rev |= 1 << (logn - 1 - j);
              }
          }
          if (i < rev) {
              std::swap(output[i], output[rev]);
          }
      }

      ICICLE_LOG_DEBUG << "AFTER BIT REVERSE (REF)";
      for (int i = 0; i < size; ++i) {
          ICICLE_LOG_DEBUG << "output[" << i << "]: " << output[i];
      }

      std::vector<S> twiddles(size / 2); 
      S omega = (dir == NTTDir::kForward) ? S::omega(logn) : S::omega_inv(logn);
      ICICLE_LOG_DEBUG << "dir: " << (dir == NTTDir::kForward ? "Forward" : "Inverse");
      twiddles[0] = S::one();
      for (int i = 1; i < size / 2; ++i) {
          twiddles[i] = twiddles[i - 1] * omega;
      }

      // NTT/INTT
      int ntt_step = 0;
      for (int len = 2; len <= size; len <<= 1) {
          ICICLE_LOG_DEBUG << "ntt_step: " << ntt_step++;
          int half_len = len / 2;
          int step = size / len;
          for (int i = 0; i < size; i += len) {
              for (int j = 0; j < half_len; ++j) {
                  S u = output[i + j];
                  S v = output[i + j + half_len] * twiddles[j * step];
                  ICICLE_LOG_DEBUG << "tw_idx=" << j * step;
                  ICICLE_LOG_DEBUG << "current_output[" << i + j << "] <-- " << output[i + j] << " + " << output[i + j + half_len] << "*" << twiddles[j * step];
                  ICICLE_LOG_DEBUG << "current_output[" << i + j + half_len << "] <-- " << output[i + j] << " - " << output[i + j + half_len] << "*" << twiddles[j * step];
                  output[i + j] = u + v;
                  output[i + j + half_len] = u - v;

              }
          }
      }
      
      if (dir == NTTDir::kInverse) {
          // Normalize results 
          S inv_size = S::inv_log_size(logn);
          for (int i = 0; i < size; ++i) {
              output[i] = output[i] * inv_size;
          }
      }

      ICICLE_LOG_DEBUG << "FINAL OUTPUT (REF)";
      for (int i = 0; i < size; ++i) {
          ICICLE_LOG_DEBUG << "output[" << i << "]: " << output[i];
      }
      
      return eIcicleError::SUCCESS;
  }


  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError bit_reverse(int size, int logn, E* output, int batch_size)
  {
      ICICLE_LOG_DEBUG << "BIT REVERSE";
      int total_size = size * batch_size;
      // ICICLE_LOG_DEBUG << "BEFORE BIT REVERSE";
      // for (int i = 0; i < total_size; ++i) {
      //     ICICLE_LOG_DEBUG << "output[" << i << "]: " << output[i];
      // }
      for (int batch = 0; batch < batch_size; ++batch) {
          E* current_output = output + batch * size;
          for (int i = 0; i < size; ++i) {
              int rev = 0;
              for (int j = 0; j < logn; ++j) {
                  if (i & (1 << j)) {
                      rev |= 1 << (logn - 1 - j);
                  }
              }
              if (i < rev) {
                  std::swap(current_output[i], current_output[rev]);
              }
          }
      }
      // ICICLE_LOG_DEBUG << "AFTER BIT REVERSE";
      // for (int i = 0; i < total_size; ++i) {
      //     ICICLE_LOG_DEBUG << "output[" << i << "]: " << output[i];
      // }

      return eIcicleError::SUCCESS;
  }

  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError cpu_ntt(const Device& device, const E* input, int size, NTTDir dir, NTTConfig<S>& config, E* output)
  {
      if (size & (size - 1)) {
          return eIcicleError::INVALID_ARGUMENT; 
      }
      
      int total_size = size * config.batch_size;
      E* temp_elements = new E[total_size];

      // ICICLE_LOG_DEBUG << "INITIAL INPUT";
      // for (int i = 0; i < total_size; ++i) {
      //     ICICLE_LOG_DEBUG << "input[" << i << "]: " << input[i];
      // }

      if (config.columns_batch) {
          // Distribute input into columns if columns_batch is set
          for (int col = 0; col < config.batch_size; ++col) {
              for (int row = 0; row < size; ++row) {
                  temp_elements[col * size + row] = input[row * config.batch_size + col];
              }
          }
      } else {
          std::copy(input, input + total_size, temp_elements);
      }
      // ICICLE_LOG_DEBUG << "INITIAL ELEMENTS ARRAY";
      // for (int i = 0; i < total_size; ++i) {
      //     ICICLE_LOG_DEBUG << "temp_elements[" << i << "]: " << temp_elements[i];
      // }
      const int logn = int(log2(size));
      
      bool dit = true;
      switch(config.ordering) { //kNN, kNR, kRN, kRR, kNM, kMN
          case Ordering::kNN:
              bit_reverse(size, logn, temp_elements, config.batch_size);
              break;
          case Ordering::kNR:
          case Ordering::kNM:
              dit = false; //dif
              break;
          case Ordering::kRR:
              bit_reverse(size, logn, temp_elements, config.batch_size);
              dit = false; //dif
              break;
          case Ordering::kRN:
          case Ordering::kMN:
              break;
          default:
              return eIcicleError::INVALID_ARGUMENT;
      }
      
      // ICICLE_LOG_DEBUG << "AFTER BIT REVERSE";
      // for (int i = 0; i < total_size; ++i) {
      //     ICICLE_LOG_DEBUG << "temp_elements[" << i << "]: " << temp_elements[i];
      // }

      S* twiddles = s_ntt_domain<S>->get_twiddles();

      // NTT/INTT
      int ntt_step = 0;
      if (dit) {
          ICICLE_LOG_DEBUG << "DIT";
          for (int batch = 0; batch < config.batch_size; ++batch) {
              E* current_temp_elements = temp_elements + batch * size;
              for (int len = 2; len <= size; len <<= 1) {
                  // ICICLE_LOG_DEBUG << "ntt_step: " << ntt_step++;
                  int half_len = len / 2;
                  int step = size / len;
                  int tw_idx = 0;
                  for (int i = 0; i < size; i += len) {
                      for (int j = 0; j < half_len; ++j) {
                          tw_idx = (dir == NTTDir::kForward)? j * step : size - j * step;
                          S u = current_temp_elements[i + j];
                          S v = current_temp_elements[i + j + half_len] * twiddles[tw_idx];
                          // ICICLE_LOG_DEBUG << "tw_idx=" << tw_idx;
                          // ICICLE_LOG_DEBUG << "current_temp_elements[" << i + j << "] <-- " << current_temp_elements[i + j] << " + " << current_temp_elements[i + j + half_len] << "*" << twiddles[tw_idx];
                          // ICICLE_LOG_DEBUG << "current_temp_elements[" << i + j + half_len << "] <-- " << current_temp_elements[i + j] << " - " << current_temp_elements[i + j + half_len] << "*" << twiddles[tw_idx];
                          current_temp_elements[i + j] = u + v;
                          current_temp_elements[i + j + half_len] = u - v;
                          // ICICLE_LOG_DEBUG << i + j << " <-- " << i + j << " + " << i + j + half_len << "*" << tw_idx;
                          // ICICLE_LOG_DEBUG << i + j + half_len << " <-- " << i + j << " - " << i + j + half_len << "*" << tw_idx;
                          
                      }
                  }
              }
          }
      } else { //dif
          ICICLE_LOG_DEBUG << "DIF";
          for (int batch = 0; batch < config.batch_size; ++batch) {
              E* current_temp_elements = temp_elements + batch * size;
              for (int len = size; len >= 2; len >>= 1) {
                  // ICICLE_LOG_DEBUG << "ntt_step: " << ntt_step++;
                  int half_len = len / 2;
                  int step = size / len;
                  int tw_idx = 0;
                  for (int i = 0; i < size; i += len) {
                      for (int j = 0; j < half_len; ++j) {
                          tw_idx = (dir == NTTDir::kForward)? j * step : size - j * step;
                          S u = current_temp_elements[i + j];
                          S v = current_temp_elements[i + j + half_len];
                          current_temp_elements[i + j] = u + v;
                          current_temp_elements[i + j + half_len] = (u - v)*twiddles[tw_idx];
                      }
                  }
              }
          }
      }

      if (dir == NTTDir::kInverse) {
          // Normalize results 
          S inv_size = S::inv_log_size(logn);
          for (int i = 0; i < total_size; ++i) {
              temp_elements[i] = temp_elements[i] * inv_size;
          }
      }

      if (config.columns_batch) {
          // Distribute output into columns if columns_batch is set
          for (int col = 0; col < config.batch_size; ++col) {
              for (int row = 0; row < size; ++row) {
                  output[row * config.batch_size + col] = temp_elements[col * size + row];
              }
          }
      } else {
          std::copy(temp_elements, temp_elements + total_size, output);
      }
      // ICICLE_LOG_DEBUG << "FINAL OUTPUT";
      // for (int i = 0; i < total_size; ++i) {
      //     ICICLE_LOG_DEBUG << "output[" << i << "]: " << output[i];
      // }
      
      delete[] temp_elements;
      return eIcicleError::SUCCESS;
  }
} // namespace ntt_template