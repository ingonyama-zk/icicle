#pragma once
#include "icicle/fields/field_config.h"

// #include <_types/_uint32_t.h>
#include <sys/types.h>

using namespace field_config;
using namespace icicle;
namespace ntt_cpu {

  /**
   * @brief Represents the coordinates of a task in the NTT hierarchy.
   * This struct holds indices that identify the position of a task within the NTT computation hierarchy.
   *
   * @param hierarchy_1_layer_idx Index of the hierarchy_1 layer.
   * @param hierarchy_1_subntt_idx Index of the sub-NTT within the hierarchy_1 layer.
   * @param hierarchy_0_layer_idx Index of the hierarchy_0 layer.
   * @param hierarchy_0_block_idx Index of the block within the hierarchy_0 layer.
   * @param hierarchy_0_subntt_idx Index of the sub-NTT within the hierarchy_0 block.
   *
   * @method bool operator==(const NttTaskCoordinates& other) const Compares two task coordinates for equality.
   */
  struct NttTaskCoordinates {
    uint32_t hierarchy_1_layer_idx = 0;
    uint32_t hierarchy_1_subntt_idx = 0;
    uint32_t hierarchy_0_layer_idx = 0;
    uint32_t hierarchy_0_block_idx = 0;
    uint32_t hierarchy_0_subntt_idx = 0;
    bool reorder = false;

    bool operator==(const NttTaskCoordinates& other) const
    {
      return hierarchy_1_layer_idx == other.hierarchy_1_layer_idx &&
             hierarchy_1_subntt_idx == other.hierarchy_1_subntt_idx &&
             hierarchy_0_layer_idx == other.hierarchy_0_layer_idx &&
             hierarchy_0_block_idx == other.hierarchy_0_block_idx &&
             hierarchy_0_subntt_idx == other.hierarchy_0_subntt_idx &&
             reorder == other.reorder;
    }

    // Default constructor
    NttTaskCoordinates() = default;

    // Constructor with parameters
    NttTaskCoordinates(uint32_t h1_layer_idx,
                       uint32_t h1_subntt_idx,
                       uint32_t h0_layer_idx,
                       uint32_t h0_block_idx,
                       uint32_t h0_subntt_idx,
                       bool reorder_flag = false)
        : hierarchy_1_layer_idx(h1_layer_idx),
          hierarchy_1_subntt_idx(h1_subntt_idx),
          hierarchy_0_layer_idx(h0_layer_idx),
          hierarchy_0_block_idx(h0_block_idx),
          hierarchy_0_subntt_idx(h0_subntt_idx),
          reorder(reorder_flag)
    {}
  };
  

  uint64_t bit_reverse(uint64_t i, uint32_t logn)
  {
    uint32_t rev = 0;
    for (uint32_t j = 0; j < logn; ++j) {
      if (i & (1 << j)) { rev |= 1 << (logn - 1 - j); }
    }
    return rev;
  };
} // namespace ntt_cpu