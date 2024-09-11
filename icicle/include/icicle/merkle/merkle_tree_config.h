#pragma once

#include "icicle/runtime.h"
#include "icicle/config_extension.h"

namespace icicle {

  /**
   * @brief Configuration structure for Merkle tree operations.
   *
   * This structure holds the configuration options for Merkle tree operations, including tree construction,
   * path computation, and verification. It allows specifying whether the data (leaves, tree, and paths)
   * reside on the device (e.g., GPU) or the host (e.g., CPU), and supports both synchronous and asynchronous
   * execution modes, as well as backend-specific extensions.
   */
  struct MerkleTreeConfig {
    icicleStreamHandle stream =
      nullptr; /**< Stream for asynchronous execution. Default is nullptr for synchronous execution. */
    bool are_leaves_on_device =
      false; /**< True if leaves are on the device (GPU), false if on the host (CPU). Default is false. */
    bool are_tree_results_on_device =
      false; /**< True if tree results are on the device (GPU), false if on the host (CPU). Default is false. */
    bool is_path_on_device =
      false; /**< True if the Merkle path is stored on the device, false if on the host. Default is false. */
    bool is_async = false;          /**< True for asynchronous execution, false for synchronous. Default is false. */
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