#include "icicle/ntt.h"
#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/log.h"
#include "icicle/fields/field_config.h"
#include "icicle/vec_ops.h"

#include <vector>
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

  public:
    static eIcicleError
    cpu_ntt_init_domain(const Device& device, const S& primitive_root, const NTTInitDomainConfig& config);
    static eIcicleError cpu_ntt_release_domain(const Device& device);
    static eIcicleError get_root_of_unity_from_domain(const Device& device, uint64_t logn, S* rou /*OUT*/);

    template <typename U, typename E>
    eIcicleError
    cpu_ntt_ref(const Device& device, const E* input, uint64_t size, NTTDir dir, NTTConfig<S>& config, E* output);

    template <typename U, typename E>
    eIcicleError
    cpu_ntt(const Device& device, const E* input, uint64_t size, NTTDir dir, NTTConfig<S>& config, E* output);

    const S* get_twiddles() const { return twiddles.get(); }
    const int get_max_size() const { return max_size; }

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

  int bit_reverse(int n, int logn)
  {
    int rev = 0;
    for (int j = 0; j < logn; ++j) {
      if (n & (1 << j)) { rev |= 1 << (logn - 1 - j); }
    }
    return rev;
  }

  template <typename E = scalar_t>
  eIcicleError reorder_by_bit_reverse(int logn, E* output, int batch_size)
  {
    uint64_t size = 1 << logn;
    for (int batch = 0; batch < batch_size; ++batch) {
      E* current_output = output + batch * size;
      int rev;
      for (int i = 0; i < size; ++i) {
        rev = bit_reverse(i, logn);
        if (i < rev) { std::swap(current_output[i], current_output[rev]); }
      }
    }
    return eIcicleError::SUCCESS;
  }

  template <typename S = scalar_t, typename E = scalar_t>
  void dit_ntt(E* elements, uint64_t size, int batch_size, const S* twiddles, NTTDir dir, int domain_max_size)
  {
    for (int batch = 0; batch < batch_size; ++batch) {
      E* current_elements = elements + batch * size;
      for (int len = 2; len <= size; len <<= 1) {
        int half_len = len / 2;
        int step = (size / len) * (domain_max_size / size);
        for (int i = 0; i < size; i += len) {
          for (int j = 0; j < half_len; ++j) {
            int tw_idx = (dir == NTTDir::kForward) ? j * step : domain_max_size - j * step;
            E u = current_elements[i + j];
            E v = current_elements[i + j + half_len] * twiddles[tw_idx];
            current_elements[i + j] = u + v;
            current_elements[i + j + half_len] = u - v;
          }
        }
      }
    }
  }

  template <typename S = scalar_t, typename E = scalar_t>
  void dif_ntt(E* elements, uint64_t size, int batch_size, const S* twiddles, NTTDir dir, int domain_max_size)
  {
    for (int batch = 0; batch < batch_size; ++batch) {
      E* current_elements = elements + batch * size;
      for (int len = size; len >= 2; len >>= 1) {
        int half_len = len / 2;
        int step = (size / len) * (domain_max_size / size);
        for (int i = 0; i < size; i += len) {
          for (int j = 0; j < half_len; ++j) {
            int tw_idx = (dir == NTTDir::kForward) ? j * step : domain_max_size - j * step;
            E u = current_elements[i + j];
            E v = current_elements[i + j + half_len];
            current_elements[i + j] = u + v;
            current_elements[i + j + half_len] = (u - v) * twiddles[tw_idx];
          }
        }
      }
    }
  }

  template <typename E = scalar_t>
  void transpose(const E* input, E* output, int rows, int cols)
  {
    for (int col = 0; col < cols; ++col) {
      for (int row = 0; row < rows; ++row) {
        output[col * rows + row] = input[row * cols + col];
      }
    }
  }

  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError coset_mul(
    int logn,
    int domain_max_size,
    E* elements,
    int batch_size,
    const S* twiddles = nullptr,
    int stride = 0,
    const std::unique_ptr<S[]>& arbitrary_coset = nullptr,
    bool bit_rev = false,
    NTTDir dir = NTTDir::kForward,
    bool columns_batch = false)
  {
    uint64_t size = 1 << logn;
    int idx;
    for (int batch = 0; batch < batch_size; ++batch) {
      E* current_elements = elements + batch * size;
      if (arbitrary_coset) {
        for (int i = 1; i < size; ++i) {
          idx = columns_batch ? batch : i;
          idx = bit_rev ? bit_reverse(idx, logn) : idx;
          current_elements[i] = current_elements[i] * arbitrary_coset[idx];
        }
      } else if (stride != 0) {
        for (int i = 1; i < size; ++i) {
          idx = bit_rev ? stride * (bit_reverse(i, logn)) : stride * i;
          idx = dir == NTTDir::kForward ? idx : domain_max_size - idx;
          current_elements[i] = current_elements[i] * twiddles[idx];
        }
      }
    }
    return eIcicleError::SUCCESS;
  }

  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError
  cpu_ntt_ref(const Device& device, const E* input, uint64_t size, NTTDir dir, NTTConfig<S>& config, E* output)
  {
    if (size & (size - 1)) {
      ICICLE_LOG_ERROR << "Size must be a power of 2. Size = " << size;
      return eIcicleError::INVALID_ARGUMENT;
    }

    // Copy input to "temp_elements" instead of pointing temp_elements to input to ensure freeing temp_elements does not
    // free the input, preventing a potential double-free error.
    // TODO [SHANIE]: Later, remove temp_elements and perform all calculations in-place
    // (implement NTT for the case where columns_batch=true, in-place).

    const uint64_t total_size = size * config.batch_size;
    auto temp_elements = std::make_unique<E[]>(total_size);
    auto vec_ops_config = default_vec_ops_config();
    if (config.columns_batch) {
      transpose(input, temp_elements.get(), size, config.batch_size);
    } else {
      std::copy(input, input + total_size, temp_elements.get());
    }
    const int logn = int(log2(size));
    const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();
    const int domain_max_size = CpuNttDomain<S>::s_ntt_domain.get_max_size();
    std::unique_ptr<S[]> arbitrary_coset = nullptr;
    int coset_stride = 0;

    if (domain_max_size < size) {
      ICICLE_LOG_ERROR << "NTT domain size is less than input size. Domain size = " << domain_max_size
                       << ", Input size = " << size;
      return eIcicleError::INVALID_ARGUMENT;
    }

    if (config.coset_gen != S::one()) { // TODO SHANIE - implement more efficient way to find coset_stride
      for (int i = 1; i <= domain_max_size; i++) {
        if (twiddles[i] == config.coset_gen) {
          coset_stride = i;
          break;
        }
      }
      if (coset_stride == 0) { // if the coset_gen is not found in the twiddles, calculate arbitrary coset
        ICICLE_LOG_DEBUG << "Coset generator not found in twiddles. Calculating arbitrary coset.";
        auto temp_cosets = std::make_unique<S[]>(domain_max_size + 1);
        arbitrary_coset = std::make_unique<S[]>(domain_max_size + 1);
        arbitrary_coset[0] = S::one();
        S coset_gen = dir == NTTDir::kForward ? config.coset_gen : S::inverse(config.coset_gen); // inverse for INTT
        for (int i = 1; i <= domain_max_size; i++) {
          arbitrary_coset[i] = arbitrary_coset[i - 1] * coset_gen;
        }
      }
    }

    bool dit = true;
    bool input_rev = false;
    bool output_rev = false;
    bool need_to_reorder = false;
    bool coset = (config.coset_gen != S::one() && dir == NTTDir::kForward);
    switch (config.ordering) { // kNN, kNR, kRN, kRR, kNM, kMN
    case Ordering::kNN:
      need_to_reorder = true;
      break;
    case Ordering::kNR:
    case Ordering::kNM:
      dit = false; // dif
      output_rev = true;
      break;
    case Ordering::kRR:
      input_rev = true;
      output_rev = true;
      need_to_reorder = true;
      dit = false; // dif
      break;
    case Ordering::kRN:
    case Ordering::kMN:
      input_rev = true;
      break;
    default:
      return eIcicleError::INVALID_ARGUMENT;
    }

    if (coset) {
      coset_mul(
        logn, domain_max_size, temp_elements.get(), config.batch_size, twiddles, coset_stride, arbitrary_coset,
        input_rev);
    }

    if (need_to_reorder) { reorder_by_bit_reverse(logn, temp_elements.get(), config.batch_size); }

    // NTT/INTT
    if (dit) {
      dit_ntt<S, E>(temp_elements.get(), size, config.batch_size, twiddles, dir, domain_max_size);
    } else {
      dif_ntt<S, E>(temp_elements.get(), size, config.batch_size, twiddles, dir, domain_max_size);
    }

    if (dir == NTTDir::kInverse) {
      // Normalize results
      S inv_size = S::inv_log_size(logn);
      for (int i = 0; i < total_size; ++i) {
        temp_elements[i] = temp_elements[i] * inv_size;
      }
      if (config.coset_gen != S::one()) {
        coset_mul(
          logn, domain_max_size, temp_elements.get(), config.batch_size, twiddles, coset_stride, arbitrary_coset,
          output_rev, dir);
      }
    }

    if (config.columns_batch) {
      transpose(temp_elements.get(), output, config.batch_size, size);
    } else {
      std::copy(temp_elements.get(), temp_elements.get() + total_size, output);
    }

    return eIcicleError::SUCCESS;
  }

  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError cpu_ntt(const Device& device, const E* input, uint64_t size, NTTDir dir, NTTConfig<S>& config, E* output)
  {
    return cpu_ntt_ref(device, input, size, dir, config, output);
  }
} // namespace ntt_cpu