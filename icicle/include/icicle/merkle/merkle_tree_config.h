#pragma once

#include "icicle/runtime.h"
#include "icicle/config_extension.h"

namespace icicle {

  /**
   * @brief Configuration struct for the Merkle tree.
   *
   * This struct is used to configure the execution of Merkle tree operations, such as building the tree,
   * computing paths, and verifying paths. It supports device-specific configurations for asynchronous
   * operations and handling data either on the device (GPU) or the host (CPU).
   */
  struct MerkleTreeConfig {
    icicleStreamHandle stream; /**< Stream for asynchronous execution. Null by default for synchronous execution. */
    bool
      are_leaves_on_device; ///< True if leaves are on the device (GPU), false if on the host (CPU). Default is false.
    bool are_tree_results_on_device; ///< True if tree results are on the device (GPU), false if on the host (CPU).
                                     ///< Default is false.
    bool is_path_on_device; ///< True if the path is stored on the device, false if on the host. Default is false.
    bool is_async; ///< True to run the Merkle tree builder asynchronously, false to run it synchronously. Default is
                   ///< false.
    ConfigExtension* ext = nullptr; ///< Backend-specific extensions for advanced configurations. Default is null.
  };

  /**
   * @brief Generates a default configuration for the Merkle tree.
   *
   * This function provides a default configuration for Merkle tree operations where the inputs, results,
   * and paths are assumed to be on the host (CPU), and the execution is synchronous by default.
   *
   * @return A default MerkleTreeConfig with synchronous execution and host-based data handling.
   */
  static MerkleTreeConfig default_merkle_tree_config()
  {
    MerkleTreeConfig config = {
      nullptr, // stream (null for synchronous execution)
      false,   // are_leaves_on_device (false for host)
      false,   // are_tree_results_on_device (false for host)
      false,   // is_path_on_device (false for host)
      false,   // is_async (false for synchronous execution)
      nullptr  // ext (no backend-specific extensions by default)
    };
    return config;
  }
} // namespace icicle