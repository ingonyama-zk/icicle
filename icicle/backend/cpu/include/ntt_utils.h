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
             hierarchy_0_subntt_idx == other.hierarchy_0_subntt_idx;
    }
  };

  uint64_t bit_reverse(uint64_t i, uint32_t logn)
  {
    uint32_t rev = 0;
    for (uint32_t j = 0; j < logn; ++j) {
      if (i & (1 << j)) { rev |= 1 << (logn - 1 - j); }
    }
    return rev;
  };

  /**
   * @brief Determines if the NTT computation should be parallelized.
   *
   * This function determines if the NTT computation should be parallelized based on the size of the NTT, the batch
   * size, and the size of the scalars.
   *
   * @param log_size The log of the size of the NTT.
   * @param log_batch_size The log of the batch size.
   * @param scalar_size The size of the scalars.
   * @return bool Returns true if the NTT computation should be parallelized, or false otherwise.
   */
  bool is_parallel(uint32_t log_size, uint32_t log_batch_size, uint32_t scalar_size)
  {
    // for small scalars, the threshold for when it is faster to use parallel NTT is higher
    if (
      (scalar_size >= 32 && (log_size + log_batch_size) <= 13) ||
      (scalar_size < 32 && (log_size + log_batch_size) <= 16)) {
      return false;
    }
    return true;
  }
} // namespace ntt_cpu