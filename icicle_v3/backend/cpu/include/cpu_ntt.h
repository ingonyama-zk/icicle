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

  int bit_reverse(int n, int logn)
  {
    int rev = 0;
    for (int j = 0; j < logn; ++j) {
      if (n & (1 << j)) { rev |= 1 << (logn - 1 - j); }
    }
    return rev;
  }

  uint64_t idx_in_mem(int element, int block_idx, int subntt_idx, const std::unique_ptr<std::vector<int>>& layers_sntt_log_size=nullptr, int layer=0)
  {
    switch (layer)
    {
      case 0:
        return block_idx + ((subntt_idx + (element << layers_sntt_log_size->at(1))) << layers_sntt_log_size->at(2));
        break;
      case 1:
        return block_idx + ((element + (subntt_idx << layers_sntt_log_size->at(1))) << layers_sntt_log_size->at(2));
        break;
      case 2:
        return (block_idx << (layers_sntt_log_size->at(1) + layers_sntt_log_size->at(2))) % (1 << (layers_sntt_log_size->at(0) + layers_sntt_log_size->at(1) + layers_sntt_log_size->at(2))) +
          ((block_idx << (layers_sntt_log_size->at(1) + layers_sntt_log_size->at(2))) / (1 << (layers_sntt_log_size->at(0) + layers_sntt_log_size->at(1) + layers_sntt_log_size->at(2))) << layers_sntt_log_size->at(2)) + element;
        break;
      default:
        ICICLE_LOG_ERROR << "Unsupported layer";
        break;
    }
    
  }

  template <typename E = scalar_t>
  eIcicleError reorder_by_bit_reverse(int logn, E* elements, int batch_size, int block_idx=0, int subntt_idx=0, const std::unique_ptr<std::vector<int>>& layers_sntt_log_size=nullptr, int layer=0)
  {

    uint64_t subntt_size = (layers_sntt_log_size==nullptr)? 1 << logn : 1 << layers_sntt_log_size->at(layer);
    int subntt_log_size = (layers_sntt_log_size==nullptr)? logn : layers_sntt_log_size->at(layer);
    uint64_t total_size =(1 << logn) * batch_size;


    for (int batch = 0; batch < batch_size; ++batch) {
      E* current_elements = elements + batch * subntt_size;
      uint64_t rev;
      for (uint64_t i = 0; i < subntt_size; ++i) {
        rev = bit_reverse(i, subntt_log_size);
        uint64_t i_mem_idx = idx_in_mem(i, block_idx, subntt_idx, layers_sntt_log_size, layer);
        uint64_t rev_mem_idx = idx_in_mem(rev, block_idx, subntt_idx, layers_sntt_log_size, layer);
        if (i < rev) {
          if (i_mem_idx < total_size && rev_mem_idx < total_size) { // Ensure indices are within bounds
            std::swap(current_elements[i_mem_idx], current_elements[rev_mem_idx]);
          } else {
            // Handle out-of-bounds error
            ICICLE_LOG_ERROR << "i=" << i << ", rev=" << rev << ", total_size=" << total_size;
            ICICLE_LOG_ERROR << "Index out of bounds: i_mem_idx=" << i_mem_idx << ", rev_mem_idx=" << rev_mem_idx;
            return eIcicleError::INVALID_ARGUMENT;
          }
        }
      }
    }
    return eIcicleError::SUCCESS;
  }

  template <typename S = scalar_t, typename E = scalar_t>
  void dit_ntt(E* elements, uint64_t total_ntt_size, int batch_size, const S* twiddles, NTTDir dir, int domain_max_size, int block_idx=0, int subntt_idx=0, const std::unique_ptr<std::vector<int>>& layers_sntt_log_size=nullptr, int layer=0) // R --> N
  {
    uint64_t subntt_size = 1 << layers_sntt_log_size->at(layer);
    for (int batch = 0; batch < batch_size; ++batch) {
      E* current_elements = elements + batch * total_ntt_size;
      for (int len = 2; len <= subntt_size; len <<= 1) {
        int half_len = len / 2;
        int step = (subntt_size / len) * (domain_max_size / subntt_size);
        for (int i = 0; i < subntt_size; i += len) {
          for (int j = 0; j < half_len; ++j) {
            int tw_idx = (dir == NTTDir::kForward) ? j * step : domain_max_size - j * step;
            uint64_t u_mem_idx = idx_in_mem(i + j, block_idx, subntt_idx, layers_sntt_log_size, layer);
            uint64_t v_mem_idx = idx_in_mem(i + j + half_len, block_idx, subntt_idx, layers_sntt_log_size, layer);
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
  void dif_ntt(E* elements, uint64_t total_ntt_size, int batch_size, const S* twiddles, NTTDir dir, int domain_max_size, int block_idx=0, int subntt_idx=0, const std::unique_ptr<std::vector<int>>& layers_sntt_log_size=nullptr, int layer=0)
  {
    uint64_t subntt_size = 1 << layers_sntt_log_size->at(layer);
    for (int batch = 0; batch < batch_size; ++batch) {
      E* current_elements = elements + batch * total_ntt_size;
      for (int len = subntt_size; len >= 2; len >>= 1) {
        int half_len = len / 2;
        int step = (subntt_size / len) * (domain_max_size / subntt_size);
        for (int i = 0; i < subntt_size; i += len) {
          for (int j = 0; j < half_len; ++j) {
            int tw_idx = (dir == NTTDir::kForward) ? j * step : domain_max_size - j * step;
            uint64_t u_mem_idx = idx_in_mem(i + j, block_idx, subntt_idx, layers_sntt_log_size, layer);
            uint64_t v_mem_idx = idx_in_mem(i + j + half_len, block_idx, subntt_idx, layers_sntt_log_size, layer);
            E u = current_elements[u_mem_idx];
            E v = current_elements[v_mem_idx];
            current_elements[u_mem_idx] = u + v;
            current_elements[v_mem_idx] = (u - v) * twiddles[tw_idx];
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

  std::unique_ptr<std::vector<int>> get_layers_subntt_log_size(int log_size)
  {
    auto layers_sntt_log_size = std::make_unique<std::vector<int>>();
    switch (log_size) {
    //ntt6-15 --> ntt16-30
    case 30:
      *layers_sntt_log_size = {15, 15, 0};
      break;
    case 29:
      *layers_sntt_log_size = {15, 14, 0};
      break;
    case 28:
      *layers_sntt_log_size = {14, 14, 0};
      break;
    case 27:
      *layers_sntt_log_size = {14, 13, 0};
      break;
    case 26:
      *layers_sntt_log_size = {13, 13, 0};
      break;
    case 25:
      *layers_sntt_log_size = {13, 12, 0};
      break;
    case 24:
      *layers_sntt_log_size = {12, 12, 0};
      break;
    case 23:
      *layers_sntt_log_size = {12, 11, 0};
      break;
    case 22:
      *layers_sntt_log_size = {11, 11, 0};
      break;
    case 21:
      *layers_sntt_log_size = {11, 10, 0};
      break;
    case 20:
      *layers_sntt_log_size = {10, 10, 0};
      break;
    case 19:
      *layers_sntt_log_size = {10, 9, 0};
      break;
    case 18:
      *layers_sntt_log_size = {9, 9, 0};
      break;
    case 17:
      *layers_sntt_log_size = {9, 8, 0};
      break;
    case 16:
      *layers_sntt_log_size = {8, 8, 0};
      break;
    //ntt3-5 --> ntt6-15
    case 15:
      *layers_sntt_log_size = {5, 5, 5};
      break;
    case 14:
      *layers_sntt_log_size = {5, 5, 4};
      break;
    case 13:
      *layers_sntt_log_size = {5, 4, 4};
      break;
    case 12:
      *layers_sntt_log_size = {4, 4, 4};
      break;
    case 11:
      *layers_sntt_log_size = {4, 4, 3};
      break;
    case 10:
      *layers_sntt_log_size = {5, 5, 0};
      break;
    case 9:
      *layers_sntt_log_size = {5, 4, 0};
      break;
    case 8:
      *layers_sntt_log_size = {4, 4, 0};
      break;
    case 7:
      *layers_sntt_log_size = {4, 3, 0};
      break;
    case 6:
      *layers_sntt_log_size = {3, 3, 0}; //debug shanie
      // *layers_sntt_log_size = {4, 2, 0};
      break;
    // ntt3-5: radix-2
    case 5:
      *layers_sntt_log_size = {5, 0, 0}; //debug shanie
      // *layers_sntt_log_size = {2, 2, 1};
      break;
    case 4:
      *layers_sntt_log_size = {4, 0, 0};
      break;
    case 3:
      *layers_sntt_log_size = {3, 0, 0};
      break;
    case 2:
      *layers_sntt_log_size = {2, 0, 0};
      break;
    case 1:
      *layers_sntt_log_size = {1, 0, 0};
      break;
    default:
      ICICLE_LOG_ERROR << "Unsupported log_size";
      break;
    }
    return layers_sntt_log_size;
  }

  template <typename S = scalar_t, typename E = scalar_t>
  void refactor_and_sort(E* layer_output, E* next_layer_input, const S* twiddles, int domain_max_size, const std::unique_ptr<std::vector<int>>& layers_sntt_log_size=nullptr, int layer=0)
  {                 
    int sntt_size = 1 << layers_sntt_log_size->at(1);
    int nof_sntts = 1 << (layers_sntt_log_size->at(0));
    int ntt_size = 1 << (layers_sntt_log_size->at(0) + layers_sntt_log_size->at(1));
    auto temp_elements = std::make_unique<E[]>(ntt_size); //FIXME - use total_memory_size instead of ntt_size - consider batch?

    // //debug shanie
    // for (int i = 0; i < nof_blocks*nof_subntts*sntt_size; i++) {
    //   ICICLE_LOG_DEBUG << "prev_layer_output[" << i << "]=" << layer_output[i];
    // }
    for (int sntt_idx = 0; sntt_idx < nof_sntts; sntt_idx++) {
      for (int elem = 0; elem < sntt_size; elem++) {
        // ICICLE_LOG_DEBUG << "block_idx=" << block_idx << ", i(sntt_idx)=" << sntt_idx << ", j(elem)=" << j; //debug shanie
        temp_elements[sntt_idx*sntt_size + elem] = layer_output[elem * nof_sntts + sntt_idx] * twiddles[(domain_max_size / ntt_size) * sntt_idx * elem];
        // ICICLE_LOG_DEBUG<< "next_layer_input[" << elem_mem_idx << "]=" << next_layer_input[elem_mem_idx] << ", layer_output[" << elem_mem_idx << "]=" << layer_output[elem_mem_idx] << ", twiddles[" << (domain_max_size / ntt_size) * sntt_idx * elem << "]=" << twiddles[(domain_max_size / ntt_size) * sntt_idx * elem];//debug shanie
      }
    }
    std::copy(temp_elements.get(), temp_elements.get() + ntt_size, next_layer_input);
    // for (int i = 0; i < nof_blocks*nof_subntts*sntt_size; i++) {
    //   ICICLE_LOG_DEBUG << "next_layer_input[" << i << "]=" << next_layer_input[i]; //debug shanie
    // }
  }

  template <typename S = scalar_t, typename E = scalar_t>
  void refactor_output(E* layer_output, E* next_layer_input, const S* twiddles, int domain_max_size, const std::unique_ptr<std::vector<int>>& layers_sntt_log_size=nullptr, int layer=0)
  {                 
    if (layer == 0 && layers_sntt_log_size->at(1)) // layer_0 -> layer_1
    {
      int subntt_size = 1 << layers_sntt_log_size->at(0);
      int nof_subntts = 1 << (layers_sntt_log_size->at(1));
      int nof_blocks = 1 << (layers_sntt_log_size->at(2));
      int ntt_size = 1 << (layers_sntt_log_size->at(0) + layers_sntt_log_size->at(1));

      // //debug shanie
      // for (int i = 0; i < nof_blocks*nof_subntts*subntt_size; i++) {
      //   ICICLE_LOG_DEBUG << "prev_layer_output[" << i << "]=" << layer_output[i];
      // }

      for (int block_idx = 0; block_idx < nof_blocks; block_idx++) {
        for (int sntt_idx = 0; sntt_idx < nof_subntts; sntt_idx++) {
          for (int elem = 0; elem < subntt_size; elem++) {
            // ICICLE_LOG_DEBUG << "block_idx=" << block_idx << ", i(sntt_idx)=" << sntt_idx << ", j(elem)=" << elem; //debug shanie
            int elem_mem_idx = idx_in_mem(elem, block_idx, sntt_idx, layers_sntt_log_size, layer);
            next_layer_input[elem_mem_idx] = layer_output[elem_mem_idx] * twiddles[(domain_max_size / ntt_size) * sntt_idx * elem];
            // ICICLE_LOG_DEBUG<< "next_layer_input[" << elem_mem_idx << "]=" << next_layer_input[elem_mem_idx] << ", layer_output[" << elem_mem_idx << "]=" << layer_output[elem_mem_idx] << ", twiddles[" << (domain_max_size / ntt_size) * sntt_idx * elem << "]=" << twiddles[(domain_max_size / ntt_size) * sntt_idx * elem];//debug shanie
          }
        }
      }
      // for (int i = 0; i < nof_blocks*nof_subntts*subntt_size; i++) {
      //   ICICLE_LOG_DEBUG << "next_layer_input[" << i << "]=" << next_layer_input[i]; //debug shanie
      // }

    }
    else if (layer == 1 && layers_sntt_log_size->at(2)) // layer_1 -> layer_2
    {
      int subntt_size = 1 << layers_sntt_log_size->at(0);
      int nof_subntts = 1 << layers_sntt_log_size->at(1);
      int nof_blocks  = 1 << layers_sntt_log_size->at(2);
      int ntt_size = 1 << (layers_sntt_log_size->at(0) + layers_sntt_log_size->at(1) + layers_sntt_log_size->at(2));
      
      for (int block_idx = 0; block_idx < nof_blocks; block_idx++) {
        for (int sntt_idx = 0; sntt_idx < nof_subntts; sntt_idx++) {
          for (int elem = 0; elem < subntt_size; elem++) {
            // ICICLE_LOG_DEBUG << ""; //debug shanie
            // ICICLE_LOG_DEBUG << "block_idx=" << block_idx << ", sntt_idx=" << sntt_idx << ", elem=" << elem; //debug shanie
            int idx_in_block = elem + sntt_idx*subntt_size;
            // ICICLE_LOG_DEBUG << "block_idx=" << block_idx << ", idx_in_block=" << idx_in_block; //debug shanie
            int elem_mem_idx = idx_in_mem(elem, block_idx, sntt_idx, layers_sntt_log_size, 0);
            // check if the index is within the bounds
            if (elem_mem_idx >= (1 << (layers_sntt_log_size->at(0) + layers_sntt_log_size->at(1) + layers_sntt_log_size->at(2)))) {
              ICICLE_LOG_ERROR << "Index out of bounds: elem_mem_idx=" << elem_mem_idx;
              return;
            }
            if ((domain_max_size / ntt_size) * block_idx * idx_in_block >= domain_max_size) {
              ICICLE_LOG_ERROR << "Index out of bounds: (domain_max_size / ntt_size) * block_idx * idx_in_block=" << (domain_max_size / ntt_size) * block_idx * idx_in_block;
              return;
            }
            // ICICLE_LOG_DEBUG << "layer_output[" << elem_mem_idx << "]=" << layer_output[elem_mem_idx] << ", twiddles[" << (domain_max_size / ntt_size) * block_idx * idx_in_block << "]=" << twiddles[(domain_max_size / ntt_size) * block_idx * idx_in_block] << ", i(block_idx)=" << block_idx << ", j(idx_in_block)=" << idx_in_block; //debug shanie;
            next_layer_input[elem_mem_idx] = layer_output[elem_mem_idx] * twiddles[(domain_max_size / ntt_size) * block_idx * idx_in_block];
            // ICICLE_LOG_DEBUG<< "next_layer_input[" << elem_mem_idx << "]=" << next_layer_input[elem_mem_idx];//debug shanie
          }
        }
      }
    }
    else
    {
      return;
    } 
  }


  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError
  sort_input(E* input, uint64_t size, int batch_size, const std::unique_ptr<std::vector<int>>& layers_sntt_log_size=nullptr){ // TODO SHANIE sort_input and sort_output can be written in a single function, but will be less readable
    if (layers_sntt_log_size==nullptr) {
      ICICLE_LOG_ERROR << "layers_sntt_log_size is null";
      return eIcicleError::INVALID_ARGUMENT;
    }
    auto temp_input = std::make_unique<E[]>(batch_size * size);
    for (int batch = 0; batch < batch_size; ++batch) {
      E* current_elements = input + batch * size;
      uint64_t idx = 0;
      uint64_t new_idx = 0;
      int cur_ntt_log_size = layers_sntt_log_size->at(0);
      int next_ntt_log_size = layers_sntt_log_size->at(1);
      for (int i = 0; i < size; i++) {
        int subntt_idx = i >> cur_ntt_log_size;
        int element = i & ((1 << cur_ntt_log_size) - 1);
        new_idx = subntt_idx + (element << next_ntt_log_size);
        temp_input[batch * size + i] = current_elements[new_idx];
      }
    }
    std::copy(temp_input.get(), temp_input.get() + batch_size * size, input);
    return eIcicleError::SUCCESS;
  }

  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError
  sort_output(E* output, uint64_t size, int batch_size, const std::unique_ptr<std::vector<int>>& layers_sntt_log_size=nullptr){
    if (layers_sntt_log_size==nullptr) {
      ICICLE_LOG_ERROR << "layers_sntt_log_size is null";
      return eIcicleError::INVALID_ARGUMENT;
    }
    auto temp_output = std::make_unique<E[]>(batch_size * size);
    for (int batch = 0; batch < batch_size; ++batch) {
      E* current_elements = output + batch * size;
      uint64_t idx = 0;
      uint64_t mem_idx = 0;
      uint64_t new_idx = 0;
      int subntt_idx;
      int element;
      int s0 = layers_sntt_log_size->at(0); 
      int s1 = layers_sntt_log_size->at(1);
      int s2 = layers_sntt_log_size->at(2);
      int p0, p1, p2;
      for (int i = 0; i < size; i++) {
        if (layers_sntt_log_size->at(2)) {
          p0 = (i >> (s1+s2));
          p1 = (((i >> s2) & ((1 << (s1)) - 1))<<s0);
          p2 = ((i & ((1 << s2) - 1))<<(s0+s1));
          new_idx = p0 + p1 + p2;
          temp_output[batch * size + new_idx] = current_elements[i];                
        }
        else {
          subntt_idx = i >> s1;
          element = i & ((1 << s1) - 1);
          new_idx = subntt_idx + (element << s0);
          temp_output[batch * size + new_idx] = current_elements[i];
        }              
      }
    }
    std::copy(temp_output.get(), temp_output.get() + batch_size * size, output);
    return eIcicleError::SUCCESS;
  }

  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError
  cpu_ntt_basic(const icicle::Device& device, const E* input, uint64_t total_ntt_size, icicle::NTTDir dir, icicle::NTTConfig<S>& config, E* output, int block_idx=0, int subntt_idx=0, const std::unique_ptr<std::vector<int>>& layers_sntt_log_size=nullptr, int layer=0)
  {
    // Copy input to "temp_elements" instead of pointing temp_elements to input to ensure freeing temp_elements does not
    // free the input, preventing a potential double-free error.
    // TODO [SHANIE]: Later, remove temp_elements and perform all calculations in-place
    // (implement NTT for the case where columns_batch=true, in-place).
    // ICICLE_LOG_DEBUG << layers_sntt_log_size->at(0) << ", " << layers_sntt_log_size->at(1) << ", " << layers_sntt_log_size->at(2); //debug shanie
    // ICICLE_LOG_DEBUG << "total_ntt_size=" << total_ntt_size << ", block_idx=" << block_idx << ", subntt_idx=" << subntt_idx << ", layer=" << layer; //debug shanie
    const uint64_t subntt_size = (1 << layers_sntt_log_size->at(layer));
    const uint64_t total_memory_size = total_ntt_size * config.batch_size;
    auto temp_elements = std::make_unique<E[]>(total_memory_size);
    if (config.columns_batch) {
      // transpose(input, temp_elements.get(), size, config.batch_size);
      ICICLE_LOG_ERROR << "columns_batch=true is not supported";
    } else {
      std::copy(input, input + total_memory_size, temp_elements.get());
    }
    //print elements
    // for (int i = 0; i < total_memory_size; i++) {
    //   ICICLE_LOG_DEBUG << "input[" << i << "]=" << input[i]; //debug shanie
    // }


    const int log_total_ntt_size = int(log2(total_ntt_size));
    const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();
    const int domain_max_size = CpuNttDomain<S>::s_ntt_domain.get_max_size();
    std::unique_ptr<S[]> arbitrary_coset = nullptr;
    int coset_stride = 0;

    if (domain_max_size < subntt_size) {
      ICICLE_LOG_ERROR << "NTT domain size is less than input size. Domain size = " << domain_max_size
                       << ", Input size = " << subntt_size;
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
    case Ordering::kNN: //dit R --> N
      need_to_reorder = true;
      break;
    case Ordering::kNR: // dif N --> R
    case Ordering::kNM: // dif N --> R
      dit = false; 
      output_rev = true;
      break;
    case Ordering::kRR:  // dif N --> R
      input_rev = true;
      output_rev = true;
      need_to_reorder = true;
      dit = false; // dif
      break;
    case Ordering::kRN: //dit R --> N
    case Ordering::kMN: //dit R --> N
      input_rev = true;
      break;
    default:
      return eIcicleError::INVALID_ARGUMENT;
    }

    if (coset) {
      coset_mul(
        log_total_ntt_size, domain_max_size, temp_elements.get(), config.batch_size, twiddles, coset_stride, arbitrary_coset,
        input_rev);
    }

    if (need_to_reorder) { reorder_by_bit_reverse(log_total_ntt_size, temp_elements.get(), config.batch_size, block_idx, subntt_idx, layers_sntt_log_size, layer); }

    // NTT/INTT
    if (dit) {
      dit_ntt<S, E>(temp_elements.get(), total_ntt_size, config.batch_size, twiddles, dir, domain_max_size, block_idx, subntt_idx, layers_sntt_log_size, layer); // R --> N
    } else {
      dif_ntt<S, E>(temp_elements.get(), total_ntt_size, config.batch_size, twiddles, dir, domain_max_size, block_idx, subntt_idx, layers_sntt_log_size, layer); // N --> R
    }

    if (dir == NTTDir::kInverse) {
      // Normalize results
      S inv_size = S::inv_log_size(log_total_ntt_size);
      for (int i = 0; i < total_memory_size; ++i) { //FIXME - normalize only the relevant elements
        temp_elements[i] = temp_elements[i] * inv_size;
      }
      if (config.coset_gen != S::one()) {
        coset_mul(
          log_total_ntt_size, domain_max_size, temp_elements.get(), config.batch_size, twiddles, coset_stride, arbitrary_coset,
          output_rev, dir);
      }
    }

    if (config.columns_batch) {
      // transpose(temp_elements.get(), output, config.batch_size, size);
      ICICLE_LOG_ERROR << "columns_batch=true is not supported";
    } else {
      std::copy(temp_elements.get(), temp_elements.get() + total_memory_size, output);
    }

    //print elements
    // for (int i = 0; i < total_memory_size; i++) {
    //   ICICLE_LOG_DEBUG << "output[" << i << "]=" << output[i]; // debug shanie
    // }

    return eIcicleError::SUCCESS;
  }

  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError
  cpu_ntt_parallel(const Device& device, int size, NTTDir dir, NTTConfig<S>& config, E* output)
  {    
    if (size & (size - 1)) {
      ICICLE_LOG_ERROR << "Size must be a power of 2. size = " << size;
      return eIcicleError::INVALID_ARGUMENT;
    }
    const int logn = int(log2(size));
    const S* twiddles = CpuNttDomain<S>::s_ntt_domain.get_twiddles();
    const int domain_max_size = CpuNttDomain<S>::s_ntt_domain.get_max_size();

    if (
      config.batch_size != 1 || config.columns_batch != false || config.coset_gen != S::one() || config.ordering != Ordering::kNN) {
      ICICLE_LOG_ERROR << "cpu_ntt_parallel: Unsupported config";
      return eIcicleError::API_NOT_IMPLEMENTED;
    }
    std::unique_ptr<std::vector<int>> layers_sntt_log_size = get_layers_subntt_log_size(logn);
    if (layers_sntt_log_size==nullptr) {
      ICICLE_LOG_ERROR << "layers_sntt_log_size is null";
      return eIcicleError::INVALID_ARGUMENT;
    }
    if (logn > 15) {
      sort_input(output, size, config.batch_size, layers_sntt_log_size);
      // for (int i=0; i<size; i++) {
      //   ICICLE_LOG_DEBUG << "input[" << i << "]=" << output[i]; //debug shanie
      // }
      // //TODO - config update if needed?
      for (int subntt_idx = 0; subntt_idx < (1<<layers_sntt_log_size->at(1)); subntt_idx++) {
        E* current_elements = output + (subntt_idx << layers_sntt_log_size->at(0)); //output + subntt_idx * subntt_size
        cpu_ntt_parallel(device, (1<<layers_sntt_log_size->at(0)), dir, config, current_elements);
      }
      refactor_and_sort<S, E>(output, output /*input fot next layer*/, twiddles, domain_max_size, layers_sntt_log_size,  0/*layer*/);
      for (int subntt_idx = 0; subntt_idx < (1<<layers_sntt_log_size->at(0)); subntt_idx++) {
        E* current_elements = output + (subntt_idx << layers_sntt_log_size->at(1)); //output + subntt_idx * subntt_size 
        cpu_ntt_parallel(device, (1<<layers_sntt_log_size->at(1)), dir, config, current_elements);
      }
      sort_output(output, size, config.batch_size, layers_sntt_log_size);
      // for (int i = 0; i < size; i++) {
      //   ICICLE_LOG_DEBUG << "final_output[" << i << "]=" << output[i]; //debug shanie
      // }

    } else {
      for (int layer = 0; layer < layers_sntt_log_size->size(); layer++) {
        if (layer == 0) {
          // for (uint64_t i=0; i<size*config.batch_size; i++) {
          //   output[i] = scalar_t::one(); //debug shanie
          // }
          // for (int i=0; i<size; i++) {
          //   ICICLE_LOG_DEBUG << "input[" << i << "]=" << output[i]; //debug shanie
          // }
          int log_nof_subntts = layers_sntt_log_size->at(1);
          int log_nof_blocks = layers_sntt_log_size->at(2);
          //TODO - config update if needed?
          for (int block_idx = 0; block_idx < (1<<log_nof_blocks); block_idx++) {
            for (int subntt_idx = 0; subntt_idx < (1<<log_nof_subntts); subntt_idx++) {
              if (layers_sntt_log_size==nullptr) {
                ICICLE_LOG_ERROR << "layers_sntt_log_size is null";
                return eIcicleError::INVALID_ARGUMENT;
              }
              else {
                // ICICLE_LOG_DEBUG << "cpu_ntt_parallel: layer=0, block_idx=" << block_idx << ", subntt_idx=" << subntt_idx; //debug shanie
                cpu_ntt_basic(device, output, size, dir, config, output, block_idx, subntt_idx, layers_sntt_log_size, layer);
                // for (int i = 0; i < size; i++) {
                //   ICICLE_LOG_DEBUG << "output[" << i << "]=" << output[i]; //debug shanie
                // }

              }
            }
          }
        }
        if (layer == 1 && layers_sntt_log_size->at(1)) {
          // for (uint64_t i=0; i<size*config.batch_size; i++) {
          //   output[i] = scalar_t::one(); //debug shanie
          // }
          // std::copy(input, input + size*config.batch_size , output);//debug shanie
          int log_nof_subntts = layers_sntt_log_size->at(0);
          int log_nof_blocks = layers_sntt_log_size->at(2);
          for (int block_idx = 0; block_idx < (1<<log_nof_blocks); block_idx++) {
            for (int subntt_idx = 0; subntt_idx < (1<<log_nof_subntts); subntt_idx++) {
              if (layers_sntt_log_size==nullptr) {
                ICICLE_LOG_ERROR << "layers_sntt_log_size is null";
                return eIcicleError::INVALID_ARGUMENT;
              }
              else {
                // ICICLE_LOG_DEBUG << "cpu_ntt_parallel: layer=1, block_idx=" << block_idx << ", subntt_idx=" << subntt_idx; //debug shanie
                cpu_ntt_basic(device, output /*input*/, size, dir, config, output, block_idx, subntt_idx, layers_sntt_log_size, layer); //input=output (in-place)
                // for (int i = 0; i < size; i++) {
                //   ICICLE_LOG_DEBUG << "output[" << i << "]=" << output[i]; //debug shanie
                // }

              }
            }
          }
        }
        if (layer == 2 && layers_sntt_log_size->at(2)) {
          // for (uint64_t i=0; i<size*config.batch_size; i++) {
          //   output[i] = scalar_t::one(); //debug shanie
          // }
          // std::copy(input, input + size*config.batch_size , output); //debug shanie
          int log_nof_blocks = layers_sntt_log_size->at(0)+layers_sntt_log_size->at(1);
          for (int block_idx = 0; block_idx < (1<<log_nof_blocks); block_idx++) {
            // ICICLE_LOG_DEBUG << "cpu_ntt_parallel: layer=2, block_idx=" << block_idx; //debug shanie
            cpu_ntt_basic(device, output /*input*/, size, dir, config, output, block_idx, 0 /*subntt_idx - not used*/, layers_sntt_log_size, layer); //input=output (in-place)
            // for (int i = 0; i < size; i++) {
            //   ICICLE_LOG_DEBUG << "output[" << i << "]=" << output[i]; //debug shanie
            // }

          }
        }
        refactor_output<S, E>(output, output /*input fot next layer*/, twiddles, domain_max_size, layers_sntt_log_size, layer);
      }
      if (layers_sntt_log_size->at(1)) { //at least 2 layers
        sort_output(output, size, config.batch_size, layers_sntt_log_size);
      }
    }
    // for (int i = 0; i < size; i++) {
    //   ICICLE_LOG_DEBUG << "final_output[" << i << "]=" << output[i]; //debug shanie
    // }
    return eIcicleError::SUCCESS;
  }

    
  template <typename S = scalar_t, typename E = scalar_t>
  eIcicleError cpu_ntt_ref(const Device& device, const E* input, uint64_t size, NTTDir dir, NTTConfig<S>& config, E* output)
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
    // for (int i = 0; i < size; i++) {
    //   ICICLE_LOG_DEBUG << "input[" << i << "]=" << input[i]; //debug shanie
    // }
    std::copy(input, input + size*config.batch_size , output);
    return cpu_ntt_parallel(device, size, dir, config, output);
    // return cpu_ntt_ref(device, input, size, dir, config, output);
  }
} // namespace ntt_cpu