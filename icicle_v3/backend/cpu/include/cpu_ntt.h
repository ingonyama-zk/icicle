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
#define PARALLEL 0

namespace ntt_cpu {

    // TODO SHANIE - after implementing real parallelism, try different sizes to choose the optimal one. Or consider using
  // a function to calculate subset sizes
  // constexpr uint32_t layers_sub_logn[31][3] = {
  //   {0, 0, 0},   {1, 0, 0},   {2, 0, 0},   {3, 0, 0},   {4, 0, 0},   {5, 0, 0},   {3, 2, 1},   {4, 3, 0},
  //   {4, 4, 0},   {5, 4, 0},   {5, 5, 0},   {4, 4, 3},   {4, 4, 4},   {5, 4, 4},   {5, 5, 4},   {5, 5, 5},
  //   {8, 8, 0},   {9, 8, 0},   {9, 9, 0},   {10, 9, 0},  {10, 10, 0}, {11, 10, 0}, {11, 11, 0}, {12, 11, 0},
  //   {12, 12, 0}, {13, 12, 0}, {13, 13, 0}, {14, 13, 0}, {14, 14, 0}, {15, 14, 0}, {15, 15, 0}};
  constexpr uint32_t layers_sub_logn[31][3] = {
    {0, 0, 0},   {1, 0, 0},   {2, 0, 0},   {3, 0, 0},   {4, 0, 0},   {5, 0, 0},   {3, 3, 0},   {4, 3, 0},
    {4, 4, 0},   {5, 4, 0},   {5, 5, 0},   {4, 4, 3},   {4, 4, 4},   {5, 4, 4},   {5, 5, 4},   {5, 5, 5},
    {8, 8, 0},   {9, 8, 0},   {9, 9, 0},   {10, 9, 0},  {10, 10, 0}, {11, 10, 0}, {11, 11, 0}, {12, 11, 0},
    {12, 12, 0}, {13, 12, 0}, {13, 13, 0}, {14, 13, 0}, {14, 14, 0}, {15, 14, 0}, {15, 15, 0}};

    struct NttTaskCordinates {
        int h1_layer_idx;
        int h1_subntt_idx;
        int h0_layer_idx;
        int h0_subntt_idx;
        int h0_block_idx;

        // Comparison operators for map
        bool operator<(const NttTaskCordinates& other) const {
            return std::tie(h1_layer_idx, h1_subntt_idx, h0_layer_idx, h0_subntt_idx, h0_block_idx) <
                std::tie(other.h1_layer_idx, other.h1_subntt_idx, other.h0_layer_idx, other.h0_subntt_idx, other.h0_block_idx);
        }
    };

struct NttTaskInfo {
    int logn; // Original size of the problem
    NttTaskCordinates c; // cordinates for the task
    std::vector<int> h0_layers_sub_logn; // Log sizes of sub-NTTs in hierarchy 0 layers
    std::vector<int> h1_layers_sub_logn; // Log sizes of sub-NTTs in hierarchy 1 layers

    // Constructor to initialize the struct
    NttTaskInfo(int logn) : logn(logn)
    {
      c.h1_layer_idx=0;
      c.h1_subntt_idx=0;
      c.h0_layer_idx=0;
      c.h0_subntt_idx=0;
      c.h0_block_idx=0;
      if (logn > 15){
        // Initialize h1_layers_sub_logn
        h1_layers_sub_logn = std::vector<int>(
          std::begin(layers_sub_logn[logn]), 
          std::end(layers_sub_logn[logn])
        );
        // Initialize h0_layers_sub_logn
        h0_layers_sub_logn = std::vector<int>(
          std::begin(layers_sub_logn[h1_layers_sub_logn[0]]), 
          std::end(layers_sub_logn[h1_layers_sub_logn[0]])
        );
      } else {
        h1_layers_sub_logn = {0, 0, 0};
        h0_layers_sub_logn = std::vector<int>(
          std::begin(layers_sub_logn[logn]), 
          std::end(layers_sub_logn[logn])
        );
      }
      ICICLE_LOG_DEBUG << "NttTaskInfo: h1_layers_sub_logn: " << h1_layers_sub_logn[0] << ", " << h1_layers_sub_logn[1] << ", " << h1_layers_sub_logn[2];
      ICICLE_LOG_DEBUG << "NttTaskInfo: h0_layers_sub_logn: " << h0_layers_sub_logn[0] << ", " << h0_layers_sub_logn[1] << ", " << h0_layers_sub_logn[2];
    }
};
  template <typename S>
  class CpuNttDomain
  {
    int max_size = 0;
    int max_log_size = 0;
    std::unique_ptr<S[]> twiddles;
    std::mutex domain_mutex;

  public:
    std::unordered_map<S, int> coset_index = {};

    static eIcicleError
    cpu_ntt_init_domain(const Device& device, const S& primitive_root, const NTTInitDomainConfig& config);
    static eIcicleError cpu_ntt_release_domain(const Device& device);
    static eIcicleError get_root_of_unity_from_domain(const Device& device, uint64_t logn, S* rou /*OUT*/);

    template <typename U, typename E>
    eIcicleError
    cpu_ntt_ref(const Device& device, const E* input, uint64_t size, NTTDir dir, const NTTConfig<S>& config, E* output);

    template <typename U, typename E>
    eIcicleError
    cpu_ntt(const Device& device, const E* input, uint64_t size, NTTDir dir, const NTTConfig<S>& config, E* output);

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

  inline uint64_t idx_in_mem(
    int element, int block_idx, int subntt_idx, const std::vector<int> layers_sntt_log_size = {}, int layer = 0)
  {
    int s0 = layers_sntt_log_size[0];
    int s1 = layers_sntt_log_size[1];
    int s2 = layers_sntt_log_size[2];
    switch (layer) {
    case 0:
      return block_idx + ((subntt_idx + (element << s1)) << s2);
    case 1:
      return block_idx + ((element + (subntt_idx << s1)) << s2);
    case 2:
      return ((block_idx << (s1 + s2)) & ((1 << (s0 + s1 + s2)) - 1)) +
             (((block_idx << (s1 + s2)) >> (s0 + s1 + s2)) << s2) + element;
    default:
      ICICLE_ASSERT(false) << "Unsupported layer";
    }
    return -1;
  }

  template <typename E = scalar_t>
  eIcicleError reorder_by_bit_reverse(
    int log_original_size,
    E* elements,
    int batch_size,
    bool columns_batch,
    int block_idx = 0,
    int subntt_idx = 0,
    std::vector<int> layers_sntt_log_size = {},
    int layer = 0)
  {
    uint64_t subntt_size = (layers_sntt_log_size.empty()) ? 1 << log_original_size : 1 << layers_sntt_log_size[layer];
    int subntt_log_size = (layers_sntt_log_size.empty()) ? log_original_size : layers_sntt_log_size[layer];
    uint64_t original_size = (1 << log_original_size);
    int stride = columns_batch ? batch_size : 1;
    for (int batch = 0; batch < batch_size; ++batch) {
      E* current_elements = columns_batch ? elements + batch : elements + batch * original_size;
      uint64_t rev;
      uint64_t i_mem_idx;
      uint64_t rev_mem_idx;
      for (uint64_t i = 0; i < subntt_size; ++i) {
        rev = bit_reverse(i, subntt_log_size);
        if (!layers_sntt_log_size.empty()) {
          i_mem_idx = idx_in_mem(i, block_idx, subntt_idx, layers_sntt_log_size, layer);
          rev_mem_idx = idx_in_mem(rev, block_idx, subntt_idx, layers_sntt_log_size, layer);
        } else {
          i_mem_idx = i;
          rev_mem_idx = rev;
        }
        if (i < rev) {
          if (i_mem_idx < original_size && rev_mem_idx < original_size) { // Ensure indices are within bounds
            std::swap(current_elements[stride * i_mem_idx], current_elements[stride * rev_mem_idx]);
          } else {
            // Handle out-of-bounds error
            ICICLE_LOG_ERROR << "i=" << i << ", rev=" << rev << ", original_size=" << original_size;
            ICICLE_LOG_ERROR << "Index out of bounds: i_mem_idx=" << i_mem_idx << ", rev_mem_idx=" << rev_mem_idx;
            return eIcicleError::INVALID_ARGUMENT;
          }
        }
      }
    }
    return eIcicleError::SUCCESS;
  }

  template <typename S = scalar_t, typename E = scalar_t>
  void dit_ntt(
    E* elements,
    uint64_t total_ntt_size,
    int batch_size,
    bool columns_batch,
    const S* twiddles,
    NTTDir dir,
    int domain_max_size,
    int block_idx = 0,
    int subntt_idx = 0,
    std::vector<int> layers_sntt_log_size = {},
    int layer = 0) // R --> N
  {
    uint64_t subntt_size = 1 << layers_sntt_log_size[layer];
    int stride = columns_batch ? batch_size : 1;
    for (int batch = 0; batch < batch_size; ++batch) {
      E* current_elements = columns_batch ? elements + batch : elements + batch * total_ntt_size;
      for (int len = 2; len <= subntt_size; len <<= 1) {
        int half_len = len / 2;
        int step = (subntt_size / len) * (domain_max_size / subntt_size);
        for (int i = 0; i < subntt_size; i += len) {
          for (int j = 0; j < half_len; ++j) {
            int tw_idx = (dir == NTTDir::kForward) ? j * step : domain_max_size - j * step;
            uint64_t u_mem_idx = stride * idx_in_mem(i + j, block_idx, subntt_idx, layers_sntt_log_size, layer);
            uint64_t v_mem_idx =
              stride * idx_in_mem(i + j + half_len, block_idx, subntt_idx, layers_sntt_log_size, layer);
            E u = current_elements[u_mem_idx];
            E v = current_elements[v_mem_idx] * twiddles[tw_idx];
            current_elements[u_mem_idx] = u + v;
            current_elements[v_mem_idx] = u - v;
          }
        }
      }
    }
  }

  template <typename S = scalar_t, typename E = scalar_t>
  void dif_ntt(
    E* elements,
    uint64_t total_ntt_size,
    int batch_size,
    bool columns_batch,
    const S* twiddles,
    NTTDir dir,
    int domain_max_size,
    int block_idx = 0,
    int subntt_idx = 0,
    std::vector<int> layers_sntt_log_size = {},
    int layer = 0)
  {
    uint64_t subntt_size = 1 << layers_sntt_log_size[layer];
    int stride = columns_batch ? batch_size : 1;
    for (int batch = 0; batch < batch_size; ++batch) {
      E* current_elements = columns_batch ? elements + batch : elements + batch * total_ntt_size;
      for (int len = subntt_size; len >= 2; len >>= 1) {
        int half_len = len / 2;
        int step = (subntt_size / len) * (domain_max_size / subntt_size);
        for (int i = 0; i < subntt_size; i += len) {
          for (int j = 0; j < half_len; ++j) {
            int tw_idx = (dir == NTTDir::kForward) ? j * step : domain_max_size - j * step;
            uint64_t u_mem_idx = stride * idx_in_mem(i + j, block_idx, subntt_idx, layers_sntt_log_size, layer);
            uint64_t v_mem_idx =
              stride * idx_in_mem(i + j + half_len, block_idx, subntt_idx, layers_sntt_log_size, layer);
            E u = current_elements[u_mem_idx];
            E v = current_elements[v_mem_idx];
            current_elements[u_mem_idx] = u + v;
            current_elements[v_mem_idx] = (u - v) * twiddles[tw_idx];
          }
        }
      }
    }
  }

  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError coset_mul(
    int logn,
    int domain_max_size,
    E* elements,
    int batch_size,
    bool columns_batch,
    const S* twiddles = nullptr,
    int stride = 0,
    const std::unique_ptr<S[]>& arbitrary_coset = nullptr,
    NTTDir dir = NTTDir::kForward)
  {
    uint64_t size = 1 << logn;
    uint64_t i_mem_idx;
    int idx;
    int batch_stride = columns_batch ? batch_size : 1;
    for (int batch = 0; batch < batch_size; ++batch) {
      E* current_elements = columns_batch ? elements + batch : elements + batch * size;
      if (arbitrary_coset) {
        for (int i = 1; i < size; ++i) {
          idx = columns_batch ? batch : i;
          current_elements[i] = current_elements[i] * arbitrary_coset[idx];
        }
      } else if (stride != 0) {
        for (int i = 1; i < size; ++i) {
          idx = stride * i;
          idx = dir == NTTDir::kForward ? idx : domain_max_size - idx;
          current_elements[batch_stride * i] = current_elements[batch_stride * i] * twiddles[idx];
        }
      }
    }
    return eIcicleError::SUCCESS;
  }

  template <typename S = scalar_t, typename E = scalar_t>
  void refactor_and_reorder(
    E* layer_output,
    E* next_layer_input,
    const S* twiddles,
    int batch_size,
    bool columns_batch,
    int domain_max_size,
    std::vector<int> layers_sntt_log_size = {},
    int layer = 0,
    icicle::NTTDir dir = icicle::NTTDir::kForward)
  {
    int sntt_size = 1 << layers_sntt_log_size[1];
    int nof_sntts = 1 << layers_sntt_log_size[0];
    int ntt_size = 1 << (layers_sntt_log_size[0] + layers_sntt_log_size[1]);
    uint64_t temp_elements_size = ntt_size * batch_size;
    auto temp_elements =
      std::make_unique<E[]>(temp_elements_size); // TODO shanie - consider using an algorithm for sorting in-place
    int stride = columns_batch ? batch_size : 1;
    for (int batch = 0; batch < batch_size; ++batch) {
      E* cur_layer_output = columns_batch ? layer_output + batch : layer_output + batch * ntt_size;
      E* cur_temp_elements = columns_batch ? temp_elements.get() + batch : temp_elements.get() + batch * ntt_size;
      for (int sntt_idx = 0; sntt_idx < nof_sntts; sntt_idx++) {
        for (int elem = 0; elem < sntt_size; elem++) {
          uint64_t tw_idx = (dir == NTTDir::kForward)
                              ? ((domain_max_size / ntt_size) * sntt_idx * elem)
                              : domain_max_size - ((domain_max_size / ntt_size) * sntt_idx * elem);
          cur_temp_elements[stride * (sntt_idx * sntt_size + elem)] =
            cur_layer_output[stride * (elem * nof_sntts + sntt_idx)] * twiddles[tw_idx];
        }
      }
    }
    std::copy(temp_elements.get(), temp_elements.get() + temp_elements_size, next_layer_input);
  }

  template <typename S = scalar_t, typename E = scalar_t>
  void refactor_subntt_output(
    E* elements,
    int batch_size,
    bool columns_batch,
    const S* twiddles,
    int domain_max_size,
    icicle::NTTDir direction = icicle::NTTDir::kForward,
    NttTaskInfo& ntt_task_info = NttTaskInfo())
  {
    int h0_subntt_size = 1 << ntt_task_info.h0_layers_sub_logn[ntt_task_info.c.h0_layer_idx];
    int h0_nof_subntts = 1 << ntt_task_info.h0_layers_sub_logn[0]; //only relevant for layer 1 
    int i, j, i_0;
    int ntt_size = ntt_task_info.c.h0_layer_idx == 0 ? 1 << (ntt_task_info.h0_layers_sub_logn[0] + ntt_task_info.h0_layers_sub_logn[1])
                                                : 1 << (ntt_task_info.h0_layers_sub_logn[0] + ntt_task_info.h0_layers_sub_logn[1] + ntt_task_info.h0_layers_sub_logn[2]);
    int stride = columns_batch ? batch_size : 1;
    uint64_t original_size = (1 << ntt_task_info.logn);
    for (int batch = 0; batch < batch_size; ++batch) {
      E* h1_subntt_elements =
      elements + stride * (ntt_task_info.c.h1_subntt_idx << ntt_task_info.h1_layers_sub_logn[ntt_task_info.c.h1_layer_idx]); // input + subntt_idx * subntt_size
      E* elements_of_current_batch = columns_batch ? h1_subntt_elements + batch : h1_subntt_elements + batch * original_size;
      for (int elem = 0; elem < h0_subntt_size; elem++) {
        uint64_t elem_mem_idx = stride * idx_in_mem(elem, ntt_task_info.c.h0_block_idx, ntt_task_info.c.h0_subntt_idx, ntt_task_info.h0_layers_sub_logn, ntt_task_info.c.h0_layer_idx);
        i = (ntt_task_info.c.h0_layer_idx == 0) ? elem : elem * h0_nof_subntts + ntt_task_info.c.h0_subntt_idx;
        j = (ntt_task_info.c.h0_layer_idx == 0) ? ntt_task_info.c.h0_subntt_idx : ntt_task_info.c.h0_block_idx;
        uint64_t tw_idx = (direction == NTTDir::kForward) ? ((domain_max_size / ntt_size) * j * i)
                                                    : domain_max_size - ((domain_max_size / ntt_size) * j * i);
        // if (ntt_task_info.c.h0_layer_idx == 1){
        //   ICICLE_LOG_DEBUG << "elem_mem_idx: " << elem_mem_idx << ", i: " << i << ", j: " << j << ", tw_idx: " << tw_idx;
        // }
        elements_of_current_batch[elem_mem_idx] = elements_of_current_batch[elem_mem_idx] * twiddles[tw_idx];
      }
    }
  }


  template <typename S = scalar_t, typename E = scalar_t>
  void refactor_output(
    E* elements,
    uint64_t tot_ntt_size,
    int batch_size,
    bool columns_batch,
    const S* twiddles,
    int domain_max_size,
    std::vector<int> layers_sntt_log_size = {},
    int layer = 0,
    icicle::NTTDir dir = icicle::NTTDir::kForward,
    NttTaskInfo& ntt_task_info = NttTaskInfo())
  {
    int subntt_size = 1 << layers_sntt_log_size[0];
    int nof_subntts = 1 << layers_sntt_log_size[1];
    int nof_blocks = 1 << layers_sntt_log_size[2];
    int i, j;
    int ntt_size = layer == 0 ? 1 << (layers_sntt_log_size[0] + layers_sntt_log_size[1])
                              : 1 << (layers_sntt_log_size[0] + layers_sntt_log_size[1] + layers_sntt_log_size[2]);
    int stride = columns_batch ? batch_size : 1;
    for (int batch = 0; batch < batch_size; ++batch) {
      E* h1_subntt_elements =
      elements + stride * (ntt_task_info.c.h1_subntt_idx << ntt_task_info.h1_layers_sub_logn[ntt_task_info.c.h1_layer_idx]); // input + subntt_idx * subntt_size
      E* current_elements = columns_batch ? h1_subntt_elements + batch : h1_subntt_elements + batch * tot_ntt_size;
      for (int block_idx = 0; block_idx < nof_blocks; block_idx++) {
        for (int sntt_idx = 0; sntt_idx < nof_subntts; sntt_idx++) {
          for (int elem = 0; elem < subntt_size; elem++) {
            uint64_t elem_mem_idx = stride * idx_in_mem(elem, block_idx, sntt_idx, layers_sntt_log_size, 0);
            i = (layer == 0) ? elem : elem + sntt_idx * subntt_size;
            j = (layer == 0) ? sntt_idx : block_idx;
            uint64_t tw_idx = (dir == NTTDir::kForward) ? ((domain_max_size / ntt_size) * j * i)
                                                        : domain_max_size - ((domain_max_size / ntt_size) * j * i);
            // if (layer == 1){
            //   ICICLE_LOG_DEBUG << "elem_mem_idx: " << elem_mem_idx << ", i: " << i << ", j: " << j << ", tw_idx: " << tw_idx << ", tw_idx: " << tw_idx;
            // }
            current_elements[elem_mem_idx] = current_elements[elem_mem_idx] * twiddles[tw_idx];
          }
        }
      }
    }
  }

  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError reorder_input(
    E* input, uint64_t size, int batch_size, bool columns_batch, const std::vector<int> layers_sntt_log_size = {})
  { // TODO shanie future - consider using an algorithm for efficient reordering
    if (layers_sntt_log_size.empty()) {
      ICICLE_LOG_ERROR << "layers_sntt_log_size is null";
      return eIcicleError::INVALID_ARGUMENT;
    }
    int stride = columns_batch ? batch_size : 1;
    auto temp_input = std::make_unique<E[]>(batch_size * size);
    for (int batch = 0; batch < batch_size; ++batch) {
      E* current_elements = columns_batch ? input + batch : input + batch * size;
      E* current_temp_input = columns_batch ? temp_input.get() + batch : temp_input.get() + batch * size;
      uint64_t idx = 0;
      uint64_t new_idx = 0;
      int cur_ntt_log_size = layers_sntt_log_size[0];
      int next_ntt_log_size = layers_sntt_log_size[1];
      for (int i = 0; i < size; i++) {
        int subntt_idx = i >> cur_ntt_log_size;
        int element = i & ((1 << cur_ntt_log_size) - 1);
        new_idx = subntt_idx + (element << next_ntt_log_size);
        current_temp_input[stride * i] = current_elements[stride * new_idx];
      }
    }
    std::copy(temp_input.get(), temp_input.get() + batch_size * size, input);
    return eIcicleError::SUCCESS;
  }

  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError reorder_output(
    E* output,
    uint64_t size,
    const std::vector<int> layers_sntt_log_size = {},
    int batch_size = 1,
    bool columns_batch = 0,
    NttTaskInfo& ntt_task_info = NttTaskInfo())
  { // TODO shanie future - consider using an algorithm for efficient reordering
    if (layers_sntt_log_size.empty()) {
      ICICLE_LOG_ERROR << "layers_sntt_log_size is null";
      return eIcicleError::INVALID_ARGUMENT;
    }
    uint64_t temp_output_size = columns_batch ? size * batch_size : size;
    auto temp_output = std::make_unique<E[]>(temp_output_size);
    uint64_t idx = 0;
    uint64_t mem_idx = 0;
    uint64_t new_idx = 0;
    int subntt_idx;
    int element;
    int s0 = layers_sntt_log_size[0];
    int s1 = layers_sntt_log_size[1];
    int s2 = layers_sntt_log_size[2];
    int p0, p1, p2;
    int stride = columns_batch ? batch_size : 1;
    int rep = columns_batch ? batch_size : 1;
    E* h1_subntt_output =
    output + stride * (ntt_task_info.c.h1_subntt_idx << ntt_task_info.h1_layers_sub_logn[ntt_task_info.c.h1_layer_idx]); // input + subntt_idx * subntt_size
    for (int batch = 0; batch < rep; ++batch) {
      E* current_elements =
        columns_batch
          ? h1_subntt_output + batch
          : h1_subntt_output; // if columns_batch=false, then output is already shifted by batch*size when calling the function
      E* current_temp_output = columns_batch ? temp_output.get() + batch : temp_output.get();
      for (int i = 0; i < size; i++) {
        if (layers_sntt_log_size[2]) {
          p0 = (i >> (s1 + s2));
          p1 = (((i >> s2) & ((1 << (s1)) - 1)) << s0);
          p2 = ((i & ((1 << s2) - 1)) << (s0 + s1));
          new_idx = p0 + p1 + p2;
          current_temp_output[stride * new_idx] = current_elements[stride * i];
        } else {
          subntt_idx = i >> s1;
          element = i & ((1 << s1) - 1);
          new_idx = subntt_idx + (element << s0);
          current_temp_output[stride * new_idx] = current_elements[stride * i];
        }
      }
    }
    std::copy(temp_output.get(), temp_output.get() + temp_output_size, h1_subntt_output);
    return eIcicleError::SUCCESS;
  }

  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError h0_cpu_ntt(
    E* input,
    uint64_t original_size,
    icicle::NTTDir direction,
    const icicle::NTTConfig<S>& config,
    NttTaskInfo& ntt_task_info)
  {
    const uint64_t subntt_size = (1 << ntt_task_info.h0_layers_sub_logn[ntt_task_info.c.h0_layer_idx]);
    const uint64_t total_memory_size = original_size * config.batch_size;
    const int log_original_size = int(log2(original_size));
    const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();
    const int domain_max_size = CpuNttDomain<S>::s_ntt_domain.get_max_size();

    int stride = config.columns_batch ? config.batch_size : 1;
    E* current_input =
      input + stride * (ntt_task_info.c.h1_subntt_idx << ntt_task_info.h1_layers_sub_logn[ntt_task_info.c.h1_layer_idx]); // input + subntt_idx * subntt_size

    reorder_by_bit_reverse(
      log_original_size, current_input, config.batch_size, config.columns_batch, ntt_task_info.c.h0_block_idx, ntt_task_info.c.h0_subntt_idx, ntt_task_info.h0_layers_sub_logn,
      ntt_task_info.c.h0_layer_idx); // TODO - check if access the fixed indexes instead of reordering may be more efficient?

    // NTT/INTT
    dit_ntt<E>(
      current_input, original_size, config.batch_size, config.columns_batch, twiddles, direction, domain_max_size, ntt_task_info.c.h0_block_idx,
      ntt_task_info.c.h0_subntt_idx, ntt_task_info.h0_layers_sub_logn, ntt_task_info.c.h0_layer_idx); // R --> N

    if (ntt_task_info.c.h0_layer_idx != 2 && ntt_task_info.h0_layers_sub_logn[ntt_task_info.c.h0_layer_idx + 1] != 0) {
      refactor_subntt_output<S, E>(
        input, config.batch_size, config.columns_batch, twiddles,
        domain_max_size, direction, ntt_task_info);
    }


    return eIcicleError::SUCCESS;
  }

  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError h1_cpu_ntt(
    uint64_t size,
    uint64_t original_size,
    NTTDir direction,
    const NTTConfig<S>& config,
    NttTaskInfo& ntt_task_info,
    E* input,
    const S* twiddles,
    const int domain_max_size = 0)
  {
    const int logn = int(log2(size));
    // Assuming that NTT fits in the cache, so we split the NTT to layers and calculate them one after the other.
    // Subntts inside the same laye are calculate in parallel.
    // Sorting is not needed, since the elements needed for each subntt are close to each other in memory.
    // Instead of sorting, we are using the function idx_in_mem to calculate the memory index of each element.
    for (ntt_task_info.c.h0_layer_idx = 0; ntt_task_info.c.h0_layer_idx < ntt_task_info.h0_layers_sub_logn.size(); ntt_task_info.c.h0_layer_idx++) {
      if (ntt_task_info.c.h0_layer_idx == 0) {
        int log_nof_subntts = ntt_task_info.h0_layers_sub_logn[1];
        int log_nof_blocks = ntt_task_info.h0_layers_sub_logn[2];
        for (ntt_task_info.c.h0_block_idx = 0; ntt_task_info.c.h0_block_idx < (1 << log_nof_blocks); ntt_task_info.c.h0_block_idx++) {
          for ( ntt_task_info.c.h0_subntt_idx = 0; ntt_task_info.c.h0_subntt_idx < (1 << log_nof_subntts); ntt_task_info.c.h0_subntt_idx++) {
            h0_cpu_ntt(input, original_size, direction, config, ntt_task_info);
          }
        }
      }
      if (ntt_task_info.c.h0_layer_idx == 1 && ntt_task_info.h0_layers_sub_logn[1]) {
        int log_nof_subntts = ntt_task_info.h0_layers_sub_logn[0];
        int log_nof_blocks = ntt_task_info.h0_layers_sub_logn[2];
        for (ntt_task_info.c.h0_block_idx = 0; ntt_task_info.c.h0_block_idx < (1 << log_nof_blocks); ntt_task_info.c.h0_block_idx++) {
          for (ntt_task_info.c.h0_subntt_idx = 0; ntt_task_info.c.h0_subntt_idx < (1 << log_nof_subntts); ntt_task_info.c.h0_subntt_idx++) {
            h0_cpu_ntt(input, original_size, direction, config, ntt_task_info); // input=output (in-place)
          }
        }
      }
      if (ntt_task_info.c.h0_layer_idx == 2 && ntt_task_info.h0_layers_sub_logn[2]) {
        int log_nof_blocks = ntt_task_info.h0_layers_sub_logn[0] + ntt_task_info.h0_layers_sub_logn[1];
        for (ntt_task_info.c.h0_block_idx = 0; ntt_task_info.c.h0_block_idx < (1 << log_nof_blocks); ntt_task_info.c.h0_block_idx++) {
          h0_cpu_ntt(input, original_size, direction, config, ntt_task_info);
        }
      }
    }
    // Sort the output at the end so that elements will be in right order.
    // TODO SHANIE  - After implementing for different ordering, maybe this should be done in a different place
    //              - When implementing real parallelism, consider sorting in parallel and in-place
    if (ntt_task_info.h0_layers_sub_logn[1]) { // at least 2 layers
      if (config.columns_batch) {
        reorder_output(input, size, ntt_task_info.h0_layers_sub_logn, config.batch_size, config.columns_batch, ntt_task_info);
      } else {
        for (int b = 0; b < config.batch_size; b++) {
          reorder_output(
            input + b * original_size, size, ntt_task_info.h0_layers_sub_logn, config.batch_size, config.columns_batch, ntt_task_info);
        }
      }
    }
    return eIcicleError::SUCCESS;
  }

  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError cpu_ntt(const Device& device, const E* input, uint64_t size, NTTDir direction, NTTConfig<S>& config, E* output)
  {
    if (size & (size - 1)) {
      ICICLE_LOG_ERROR << "Size must be a power of 2. size = " << size;
      return eIcicleError::INVALID_ARGUMENT;
    }
    const int logn = int(log2(size));
    const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();
    const int domain_max_size = CpuNttDomain<S>::s_ntt_domain.get_max_size();
    NttTaskInfo ntt_task_info(logn);

    int coset_stride = 0;
    std::unique_ptr<S[]> arbitrary_coset = nullptr;
    if (config.coset_gen != S::one()) { // TODO SHANIE - implement more efficient way to find coset_stride
      try {
        coset_stride = CpuNttDomain<S>::s_ntt_domain.coset_index.at(config.coset_gen); //Coset generator found in twiddles
      } catch (const std::out_of_range& oor) { //Coset generator not found in twiddles. Calculating arbitrary coset
        arbitrary_coset = std::make_unique<S[]>(domain_max_size + 1);
        arbitrary_coset[0] = S::one();
        S coset_gen = direction == NTTDir::kForward ? config.coset_gen : S::inverse(config.coset_gen); // inverse for INTT
        for (int i = 1; i <= domain_max_size; i++) {
          arbitrary_coset[i] = arbitrary_coset[i - 1] * coset_gen;
        }
      }
    }
    uint64_t total_memory_size = size * config.batch_size;
    std::copy(input, input + total_memory_size, output);
    if (config.ordering == Ordering::kRN || config.ordering == Ordering::kRR) {
      reorder_by_bit_reverse(
        logn, output, config.batch_size,
        config.columns_batch); // TODO - check if access the fixed indexes instead of reordering may be more efficient?
    }

    if (config.coset_gen != S::one() && direction == NTTDir::kForward) {
      coset_mul(
        logn, domain_max_size, output, config.batch_size, config.columns_batch, twiddles, coset_stride, arbitrary_coset,
        direction);
    }

    if (logn > 15) {
      // TODO future - maybe can start 4'rth layer in parallel to 3'rd layer?
      // Assuming that NTT doesn't fit in the cache, so we split the NTT to 2 layers and calculate them one after the
      // other. Inside each layer each sub-NTT calculation is split to layers as well, and those are calculated in
      // parallel. Sorting is done between the layers, so that the elements needed for each sunbtt are close to each
      // other in memory.

      reorder_input(output, size, config.batch_size, config.columns_batch, ntt_task_info.h1_layers_sub_logn);
      // std::cout << "PRINT REORDERED INPUT";
      // for (int i = 0; i < size; i++) {
      //   std::cout <<"output["<<i<<"] = "<< output[i] << std::endl;
      // }

      int log_nof_h1_subntts_todo_in_parallel = 15 - ntt_task_info.h1_layers_sub_logn[0] - int(log2(config.batch_size));
      int nof_h1_subntts_todo_in_parallel = 1 << log_nof_h1_subntts_todo_in_parallel;
      int log_nof_subntts_chunks = ntt_task_info.h1_layers_sub_logn[1] - log_nof_h1_subntts_todo_in_parallel;
      int nof_subntts_chunks = 1 << log_nof_subntts_chunks;
      for (int h1_subntts_chunck_idx = 0; h1_subntts_chunck_idx < nof_subntts_chunks; h1_subntts_chunck_idx++) {
        for (int h1_subntt_idx_in_chunck = 0; h1_subntt_idx_in_chunck < nof_h1_subntts_todo_in_parallel; h1_subntt_idx_in_chunck++) {
          ntt_task_info.c.h1_subntt_idx =  h1_subntts_chunck_idx*nof_h1_subntts_todo_in_parallel + h1_subntt_idx_in_chunck;

          h1_cpu_ntt((1 << ntt_task_info.h1_layers_sub_logn[0]), size, direction, config, ntt_task_info, output, twiddles, domain_max_size);
          // ICICLE_LOG_DEBUG << "AFTER LAYER_0 H1_SUBNTT_IDX: "<< h1_subntt_idx;
          // for (int i = 0; i < size; i++) {
          //   ICICLE_LOG_DEBUG << output[i];
          // }
        }
      }

      refactor_and_reorder<S, E>(
        output, output /*input for next layer*/, twiddles, config.batch_size, config.columns_batch, domain_max_size,
        ntt_task_info.h1_layers_sub_logn, 0 /*layer*/, direction);
      // std::cout << "AFTER refactor_and_reorder";
      // for (int i = 0; i < size; i++) {
      //   std::cout <<"output["<<i<<"] = "<< output[i] << std::endl;
      // }
      ntt_task_info.c.h1_layer_idx = 1;
      ntt_task_info.h0_layers_sub_logn = std::vector<int>(
        std::begin(layers_sub_logn[ntt_task_info.h1_layers_sub_logn[1]]), 
        std::end(layers_sub_logn[ntt_task_info.h1_layers_sub_logn[1]]));

      log_nof_h1_subntts_todo_in_parallel = 15 - ntt_task_info.h1_layers_sub_logn[1] - int(log2(config.batch_size));
      nof_h1_subntts_todo_in_parallel = 1 << log_nof_h1_subntts_todo_in_parallel;
      log_nof_subntts_chunks = ntt_task_info.h1_layers_sub_logn[0] - log_nof_h1_subntts_todo_in_parallel;
      nof_subntts_chunks = 1 << log_nof_subntts_chunks;
      for (int h1_subntts_chunck_idx = 0; h1_subntts_chunck_idx < nof_subntts_chunks; h1_subntts_chunck_idx++) {
        for (int h1_subntt_idx_in_chunck = 0; h1_subntt_idx_in_chunck < nof_h1_subntts_todo_in_parallel; h1_subntt_idx_in_chunck++) {
          ntt_task_info.c.h1_subntt_idx =  h1_subntts_chunck_idx*nof_h1_subntts_todo_in_parallel + h1_subntt_idx_in_chunck;
          h1_cpu_ntt((1 << ntt_task_info.h1_layers_sub_logn[1]), size, direction, config, ntt_task_info, output, twiddles, domain_max_size);
        }
      }
      // std::cout << "AFTER LAYER_1";
      // for (int i = 0; i < size; i++) {
      //   std::cout <<"output["<<i<<"] = "<< output[i] << std::endl;
      // }

      ntt_task_info.c.h1_subntt_idx = 0; // reset so that refactor_output will calculate the correct memory index
      if (config.columns_batch) {
        reorder_output(output, size, ntt_task_info.h1_layers_sub_logn, config.batch_size, config.columns_batch, ntt_task_info);
      } else {
        for (int b = 0; b < config.batch_size; b++) {
          reorder_output(output + b * size, size, ntt_task_info.h1_layers_sub_logn, config.batch_size, config.columns_batch, ntt_task_info);
        }
      }
      // ICICLE_LOG_DEBUG << "AFTER reorder_output";
      // for (int i = 0; i < size; i++) {
      //   ICICLE_LOG_DEBUG << output[i];
      // }
    } else {
      h1_cpu_ntt(size, size, direction, config, ntt_task_info, output, twiddles, domain_max_size);
    }

    if (direction == NTTDir::kInverse) { // TODO SHANIE - do that in parallel
      S inv_size = S::inv_log_size(logn);
      for (uint64_t i = 0; i < total_memory_size; ++i) {
        output[i] = output[i] * inv_size;
      }
      if (config.coset_gen != S::one()) {
        // bool output_rev = config.ordering == Ordering::kNR || config.ordering == Ordering::kNM || config.ordering ==
        // Ordering::kRR;
        bool output_rev = false;
        coset_mul(
          logn, domain_max_size, output, config.batch_size, config.columns_batch, twiddles, coset_stride,
          arbitrary_coset, direction);
      }
    }

    if (config.ordering == Ordering::kNR || config.ordering == Ordering::kRR) {
      reorder_by_bit_reverse(
        logn, output, config.batch_size,
        config.columns_batch); // TODO - check if access the fixed indexes instead of reordering may be more efficient?
    }
    // std::cout << "FINAL OUTPUT";
    // for (int i = 0; i < size; i++) {
    //   std::cout <<"output["<<i<<"] = "<< output[i] << std::endl;
    // }

    return eIcicleError::SUCCESS;
  }
} // namespace ntt_cpu















  // template <typename S = scalar_t, typename E = scalar_t>
  // eIcicleError cpu_ntt_parallel_try(
  //   uint64_t size,
  //   uint64_t original_size,
  //   NTTDir dir,
  //   const NTTConfig<S>& config,
  //   E* output,
  //   const S* twiddles,
  //   const int domain_max_size = 0)
  // {
  //   const int logn = int(log2(size));
  //   std::vector<int> layers_sntt_log_size(
  //     std::begin(layers_subntt_log_size[logn]), std::end(layers_subntt_log_size[logn]));

  //   unsigned int max_nof_parallel_threads = std::thread::hardware_concurrency();
  //   std::cout << "Number of concurrent threads supported: " << max_nof_parallel_threads << std::endl;

  //   std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> start_times(max_nof_parallel_threads*3);
  //   std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> finish_times(max_nof_parallel_threads*3);
  //   // Assuming that NTT fits in the cache, so we split the NTT to layers and calculate them one after the other.
  //   // Subntts inside the same laye are calculate in parallel.
  //   // Sorting is not needed, since the elements needed for each subntt are close to each other in memory.
  //   // Instead of sorting, we are using the function idx_in_mem to calculate the memory index of each element.
  //   for (int layer = 0; layer < layers_sntt_log_size.size(); layer++) {
  //     #if PARALLEL
  //     std::vector<std::thread> threads;
  //     #endif
  //     if (layer == 0) {
  //       int nof_subntts =  1 << layers_sntt_log_size[1];
  //       int nof_blocks = 1 << layers_sntt_log_size[2];
  //       int subntts_per_thread = (nof_subntts + max_nof_parallel_threads - 1) / max_nof_parallel_threads;
  //       int nof_threads = nof_subntts / subntts_per_thread;
  //       ICICLE_LOG_DEBUG << "layer: "<< layer <<", number of threads: " << nof_threads << ", subntts per thread: " << subntts_per_thread << ", nof_subntts: " << nof_subntts << ", nof_blocks: " << nof_blocks;
  //       for (int block_idx = 0; block_idx < nof_blocks; block_idx++) {
  //         for (int thread_idx = 0; thread_idx < nof_threads; thread_idx++) {
  //           #if PARALLEL
  //           threads.emplace_back([&, block_idx, thread_idx] {
  //           #endif
  //             for (int subntt_idx = thread_idx; subntt_idx < nof_subntts; subntt_idx += nof_threads) {
  //               int idx = block_idx * nof_subntts + subntt_idx;
  //               ICICLE_LOG_DEBUG << "layer: "<< layer <<", block_idx: " << block_idx << ", subntt_idx: " << subntt_idx << ", idx: " << idx;
  //               start_times[idx] = std::chrono::high_resolution_clock::now();
  //               cpu_ntt_basic(
  //                 output, original_size, dir, config, output, block_idx, subntt_idx, layers_sntt_log_size, layer);
  //               finish_times[idx] = std::chrono::high_resolution_clock::now();
  //             }
  //           #if PARALLEL
  //           });
  //           #endif
  //         }
  //       }
  //       #if PARALLEL
  //       for (auto& th : threads) {
  //         th.join();
  //       }
  //       #endif
  //     }
  //     if (layer == 1 && layers_sntt_log_size[1]) {
  //       int nof_subntts = 1 << layers_sntt_log_size[0];
  //       int nof_blocks = 1 << layers_sntt_log_size[2];
  //       int subntts_per_thread = (nof_subntts + max_nof_parallel_threads - 1) / max_nof_parallel_threads;
  //       int nof_threads = nof_subntts / subntts_per_thread;
  //       ICICLE_LOG_DEBUG << "layer: "<< layer <<", number of threads: " << nof_threads << ", subntts per thread: " << subntts_per_thread << ", nof_subntts: " << nof_subntts << ", nof_blocks: " << nof_blocks;
  //       for (int block_idx = 0; block_idx < nof_blocks; block_idx++) {
  //         for (int thread_idx = 0; thread_idx < nof_threads; thread_idx++) {
  //           #if PARALLEL
  //           threads.emplace_back([&, block_idx, thread_idx] {
  //           #endif
  //             for (int subntt_idx = thread_idx; subntt_idx < nof_subntts; subntt_idx += nof_threads) {
  //               int idx = (1 << layers_sntt_log_size[1]) + block_idx * nof_subntts + subntt_idx;
  //               ICICLE_LOG_DEBUG << "layer: "<< layer <<", block_idx: " << block_idx << ", subntt_idx: " << subntt_idx << ", idx: " << idx;
  //               start_times[idx] = std::chrono::high_resolution_clock::now();
  //               cpu_ntt_basic(
  //                 output, original_size, dir, config, output, block_idx, subntt_idx, layers_sntt_log_size, layer);
  //               finish_times[idx] = std::chrono::high_resolution_clock::now();
  //             }
  //           #if PARALLEL
  //           });
  //           #endif
  //         }
  //       }
  //       #if PARALLEL
  //       for (auto& th : threads) {
  //         th.join();
  //       }
  //       #endif
  //     }
  //     if (layer == 2 && layers_sntt_log_size[2]) {
  //       int nof_blocks = 1 << (layers_sntt_log_size[0] + layers_sntt_log_size[1]);
  //       int subntts_per_thread = (nof_blocks + max_nof_parallel_threads - 1) / max_nof_parallel_threads;
  //       int nof_threads = nof_blocks / subntts_per_thread;
  //       for (int thread_idx = 0; thread_idx < nof_threads; thread_idx++) {
  //         #if PARALLEL
  //         threads.emplace_back([&, thread_idx] {
  //         #endif
  //           for (int block_idx = thread_idx; block_idx < nof_blocks; block_idx += nof_threads) {
  //             int idx = (1 << layers_sntt_log_size[1]) + (1 << layers_sntt_log_size[0]) + block_idx;
  //             start_times[idx] = std::chrono::high_resolution_clock::now();
  //             cpu_ntt_basic(
  //               output, original_size, dir, config, output, block_idx, 0/*subntt_idx - not used*/, layers_sntt_log_size, layer);
  //             finish_times[idx] = std::chrono::high_resolution_clock::now();
  //           }
  //         #if PARALLEL
  //         });
  //         #endif
  //       }
  //       #if PARALLEL
  //       for (auto& th : threads) {
  //         th.join();
  //       }
  //       #endif
  //     }
  //     // if (layer != 2 && layers_sntt_log_size[layer + 1] != 0) {
  //     //   refactor_output<S, E>(
  //     //     output, original_size, config.batch_size, config.columns_batch, twiddles,
  //     //     domain_max_size, layers_sntt_log_size, layer, dir);
  //     // }
  //   }
  //   // Sort the output at the end so that elements will be in right order.
  //   // TODO SHANIE  - After implementing for different ordering, maybe this should be done in a different place
  //   //              - When implementing real parallelism, consider sorting in parallel and in-place
  //   if (layers_sntt_log_size[1]) { // at least 2 layers
  //     if (config.columns_batch) {
  //       reorder_output(output, size, layers_sntt_log_size, config.batch_size, config.columns_batch);
  //     } else {
  //       for (int b = 0; b < config.batch_size; b++) {
  //         reorder_output(
  //           output + b * original_size, size, layers_sntt_log_size, config.batch_size, config.columns_batch);
  //       }
  //     }
  //   }
  //   // #if PARALLEL
  //   // Print start and finish times
  //   for (size_t i = 0; i < start_times.size(); ++i) {
  //       auto start_ns = std::chrono::duration_cast<std::chrono::microseconds>(start_times[i].time_since_epoch()).count();
  //       auto finish_ns = std::chrono::duration_cast<std::chrono::microseconds>(finish_times[i].time_since_epoch()).count();
  //       // std::cout << "thread " << i << " started at " << start_ns
  //       //   << " μs and finished at " << finish_ns << " μs. Total: " << (finish_ns - start_ns) << " μs\n";
  //   }
  //   // #endif
  //   return eIcicleError::SUCCESS;
  // }
  
