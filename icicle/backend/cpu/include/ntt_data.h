#pragma once
#include "icicle/utils/log.h"
#include "tasks_manager.h"
#include "cpu_ntt_domain.h"

// #include <_types/_uint32_t.h>
#include <csetjmp>
#include <cstdint>
#include <sys/types.h>
#include <deque>
#include <functional>
#include <unordered_map>

#define HIERARCHY_1 22

namespace ntt_cpu {

  /**
   * @brief Defines the log sizes of sub-NTTs for different problem sizes.
   *
   * `layers_sub_logn` specifies the log sizes for up to three layers (hierarchy1 or hierarchy0) in the NTT computation.
   * - The outer index represents the log size (`logn`) of the original NTT problem.
   * - Each inner array contains three integers corresponding to the log sizes for each hierarchical layer.
   *
   * Example: `layers_sub_logn[14] = {14, 13, 0}` means for `logn = 14`, the sub-NTT log sizes are 14 for the first
   * layer, 13 for the second, and 0 for the third.
   */
  constexpr uint32_t layers_sub_logn[31][3] = {
    {0, 0, 0},   {1, 0, 0},   {2, 0, 0},   {3, 0, 0},   {4, 0, 0},   {5, 0, 0},   {3, 3, 0},  {2, 5, 0},
    {3, 5, 0},   {4, 5, 0},   {5, 5, 0},   {5, 6, 0},   {5, 7, 0},   {5, 8, 0},   {5, 9, 0},  {5, 10, 0},
    {5, 5, 6},   {5, 5, 7},   {5, 5, 8},   {5, 5, 9},   {5, 5, 10},  {5, 5, 11},  {5, 5, 12}, {12, 11, 0},
    {12, 12, 0}, {13, 12, 0}, {13, 13, 0}, {14, 13, 0}, {14, 14, 0}, {15, 14, 0}, {15, 15, 0}};

  /**
   * @brief Represents the log sizes of sub-NTTs in the NTT computation hierarchy.
   *
   * This struct stores the log sizes of the sub-NTTs for both hierarchy_0 and hierarchy_1  layers,
   * based on the overall log size (`logn`) of the NTT problem.
   *
   * @param logn The log size of the entire NTT problem.
   * @param size The size of the NTT problem, calculated as `1 << logn`.
   * @param hierarchy_0_layers_sub_logn Log sizes of sub-NTTs for hierarchy_0 layers.
   * @param hierarchy_1_layers_sub_logn Log sizes of sub-NTTs for hierarchy_1 layers.
   *
   * @method NttSubHierarchies(uint32_t logn) Initializes the struct based on the given `logn`.
   */
  struct NttSubHierarchies {
    std::vector<std::vector<uint32_t>> hierarchy_0_layers_sub_logn; // Log sizes of sub-NTTs in hierarchy 0 layers
    std::vector<uint32_t> hierarchy_1_layers_sub_logn;              // Log sizes of sub-NTTs in hierarchy 1 layers

    // Constructor to initialize the struct
    NttSubHierarchies(uint32_t logn)
    {
      if (logn > HIERARCHY_1) {
        // Initialize hierarchy_1_layers_sub_logn
        hierarchy_1_layers_sub_logn =
          std::vector<uint32_t>(std::begin(layers_sub_logn[logn]), std::end(layers_sub_logn[logn]));
        // Initialize hierarchy_0_layers_sub_logn
        hierarchy_0_layers_sub_logn = {
          std::vector<uint32_t>(
            std::begin(layers_sub_logn[hierarchy_1_layers_sub_logn[0]]),
            std::end(layers_sub_logn[hierarchy_1_layers_sub_logn[0]])),
          std::vector<uint32_t>(
            std::begin(layers_sub_logn[hierarchy_1_layers_sub_logn[1]]),
            std::end(layers_sub_logn[hierarchy_1_layers_sub_logn[1]]))};
      } else {
        hierarchy_1_layers_sub_logn = {0, 0, 0};
        hierarchy_0_layers_sub_logn = {
          std::vector<uint32_t>(std::begin(layers_sub_logn[logn]), std::end(layers_sub_logn[logn])), {0, 0, 0}};
      }
    }
  };

  template <typename S = scalar_t, typename E = scalar_t>
  struct NttData {
    const uint32_t logn;                         // log of the original NTT size.
    const uint32_t size;                         // Size of the original NTT problem.
    const NttSubHierarchies ntt_sub_hierarchies; // Log sizes of sub-NTTs based on the original NTT log size.
    E* const elements;                           // Pointer to the output elements array.
    const NTTConfig<S>& config;                  // Configuration settings for the NTT computation.
    const NTTDir direction;                      // Direction of the NTT computation (forward or inverse).
    const bool is_parallel;                      // Flag indicating if the NTT computation is parallel.
    const uint32_t nof_elems_per_cacheline;            // Number of elements per cacheline
    uint32_t coset_stride = 0; // Stride value for coset multiplication, retrieved from the NTT domain.
    std::unique_ptr<S[]> arbitrary_coset = nullptr; // Array holding arbitrary coset values if needed.
    NttData(uint32_t logn, E* elements, const NTTConfig<S>& config, NTTDir direction, bool is_parallel, const uint32_t nof_elems_per_cacheline)
        : logn(logn), size(1 << logn), ntt_sub_hierarchies(logn), elements(elements), config(config),
          direction(direction), is_parallel(is_parallel), nof_elems_per_cacheline(nof_elems_per_cacheline)
    {
      if (config.coset_gen != S::one()) {
        try {
          coset_stride =
            CpuNttDomain<S>::s_ntt_domain.get_coset_stride(config.coset_gen); // Coset generator found in twiddles
        } catch (const std::out_of_range& oor) { // Coset generator not found in twiddles. Calculating arbitrary coset
          int domain_max_size = CpuNttDomain<S>::s_ntt_domain.get_max_size();
          arbitrary_coset = std::make_unique<S[]>(domain_max_size + 1);
          arbitrary_coset[0] = S::one();
          S coset_gen =
            direction == NTTDir::kForward ? config.coset_gen : S::inverse(config.coset_gen); // inverse for INTT
          for (uint32_t i = 1; i <= CpuNttDomain<S>::s_ntt_domain.get_max_size(); i++) {
            arbitrary_coset[i] = arbitrary_coset[i - 1] * coset_gen;
          }
        }
      }
    }
  };
} // namespace ntt_cpu