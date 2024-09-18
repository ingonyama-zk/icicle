#pragma once
#include "icicle/utils/log.h"
#include "tasks_manager.h"
#include "cpu_ntt_domain.h"

#include <_types/_uint32_t.h>
#include <sys/types.h>
#include <deque>
#include <functional>
#include <unordered_map>

#define HIERARCHY_1 15

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
    {0, 0, 0},   {1, 0, 0},   {2, 0, 0},   {3, 0, 0},   {4, 0, 0},   {5, 0, 0},   {3, 3, 0},   {4, 3, 0},
    {4, 4, 0},   {5, 4, 0},   {5, 5, 0},   {4, 4, 3},   {4, 4, 4},   {5, 4, 4},   {5, 5, 4},   {5, 5, 5},
    {8, 8, 0},   {9, 8, 0},   {9, 9, 0},   {10, 9, 0},  {10, 10, 0}, {11, 10, 0}, {11, 11, 0}, {12, 11, 0},
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
   * @method NttSubLogn(uint32_t logn) Initializes the struct based on the given `logn`.
   */
  struct NttSubLogn {
    uint32_t logn;                                                  // Original log_size of the problem
    uint64_t size;                                             // Original size of the problem
    std::vector<std::vector<uint32_t>> hierarchy_0_layers_sub_logn; // Log sizes of sub-NTTs in hierarchy 0 layers
    std::vector<uint32_t> hierarchy_1_layers_sub_logn;              // Log sizes of sub-NTTs in hierarchy 1 layers

    // Constructor to initialize the struct
    NttSubLogn(uint32_t logn) : logn(logn)
    {
      size = 1 << logn;
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
    const NttSubLogn ntt_sub_logn;
    E* const elements;
    const NTTConfig<S>& config;
    const NTTDir direction;
    NttData(uint32_t logn, E* elements, const NTTConfig<S>& config, NTTDir direction)
        : ntt_sub_logn(logn), elements(elements), config(config), direction(direction) {
        }
  };

} // namespace ntt_cpu