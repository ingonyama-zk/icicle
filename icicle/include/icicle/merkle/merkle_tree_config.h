#pragma once

#include "icicle/runtime.h"
#include "icicle/config_extension.h"

namespace icicle {

  /**
   * @brief Enum representing the padding policy when the input is smaller than expected by the tree structure.
   */
  enum class PaddingPolicy {
    None,        /**< No padding, assume input is correctly sized. */
    ZeroPadding, /**< Pad the input with zeroes to fit the expected input size. */
    LastValue    /**< Pad the input by repeating the last value. */
  };

  /**
   * @brief Configuration structure for Merkle tree operations.
   *
   * This structure holds the configuration options for Merkle tree operations, including tree construction,
   * path computation, and verification. It allows specifying whether the data (leaves, tree, and paths)
   * reside on the device (e.g., GPU) or the host (e.g., CPU), and supports both synchronous and asynchronous
   * execution modes, as well as backend-specific extensions. It also provides a padding policy for handling
   * cases where the input size is smaller than expected by the tree structure.
   */
  struct MerkleTreeConfig {
    icicleStreamHandle stream =
      nullptr; /**< Stream for asynchronous execution. Default is nullptr for synchronous execution. */
    bool is_leaves_on_device =
      false; /**< True if leaves are on the device (GPU), false if on the host (CPU). Default is false. */
    bool is_tree_on_device = false; /**< True if the tree results are allocated on the device (GPU), false if on the
                                                                       host (CPU). Default is false. */
    bool is_async = false;          /**< True for asynchronous execution, false for synchronous. Default is false. */
    PaddingPolicy padding_policy =
      PaddingPolicy::None;          /**< Policy for handling cases where the input is smaller than expected. */
    ConfigExtension* ext = nullptr; /**< Backend-specific extensions for advanced configurations. Default is nullptr. */
  };

  /**
   * @brief Generates a default configuration for Merkle tree operations.
   *
   * This function provides a default configuration for Merkle tree operations with synchronous execution
   * and all data (leaves, tree results, and paths) residing on the host (CPU).
   *
   * @return A default MerkleTreeConfig with host-based execution and no backend-specific extensions.
   */
  static MerkleTreeConfig default_merkle_tree_config() { return MerkleTreeConfig(); }

} // namespace icicle