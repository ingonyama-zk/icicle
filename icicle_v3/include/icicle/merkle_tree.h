#pragma once

#include "errors.h"
#include "runtime.h"
#include "hash.h"
#include "icicle/utils/utils.h"

#include <cstdint>
#include <functional>


/**
 * @brief Configuration struct for the merkle tree.
 */
struct MerkleTreeConfig {
    bool are_leaves_on_device = false;  ///< True if inputs are on device, false if on host. Default is false.
    bool are_tree_results_on_device = false; ///< True if outputs are on device, false if on host. Default is false.
    bool is_path_on_device = false; ///< True if outputs are on device, false if on host. Default is false.
    bool is_async = false;              ///< True to run the tree builder asynchronously, false to run it synchronously. Default is false.
    ConfigExtension* ext = nullptr; ///< Backend-specific extensions.
};

/**
 * @brief Class representing a Merkle tree.
 */
class MerkleTree {
 public:
    const unsigned int nof_layers;             ///< Number of layers in the tree.
    const unsigned int nof_limbs_in_leaf;             ///< Number of layers in the tree.
    const Hash **layer_hashes;                  ///< Array of hashes for each layer (first for leaves, last for root).
    const unsigned int output_store_min_layer; ///< Minimum layer index to store in the output (start index = 0, max_index = nof_layers - 1).
    const unsigned int output_store_max_layer; ///< Maximum layer index to store in the output (start index = 0, max_index = nof_layers - 1).

    /**
     * @brief Constructor for the MerkleTree class.
     * @param nof_layers Number of layers in the tree.
     * @param layer_hashes Array of pointers to hashes for each layer.
     * @param output_store_min_layer Minimum layer index to store in the output.
     * @param output_store_max_layer Maximum layer index to store in the output.
     */
    MerkleTree(unsigned int nof_layers, const Hash **layer_hashes,
               unsigned int output_store_min_layer, unsigned int output_store_max_layer)
          : nof_layers{nof_layers}, layer_hashes{layer_hashes}, 
          output_store_min_layer{output_store_min_layer}, output_store_max_layer{output_store_max_layer} {}

    /**
     * @brief Pure virtual function to allocate the tree results.
     * @param tree_results Pointer to the tree results pointer.
     * @param bytes Size of the tree results allocated.
     * @param config Configuration struct for the merkle tree.
     */
    virtual eIcicleError allocate_tree_results(limb_t **tree_results, unsigned& bytes /*OUT*/, const MerkleTreeConfig& config) const = 0;
    
    /**
     * @brief Pure virtual function to allocate the path.
     * @param path Pointer to the path pointer.
     * @param bytes Size of the path allocated.
     * @param config Configuration struct for the merkle tree.
     */
    virtual eIcicleError allocate_path(limb_t **path, unsigned& bytes /*OUT*/, const MerkleTreeConfig& config) const = 0;

    /**
     * @brief Pure virtual function to build the Merkle tree from the leaves.
     * @param leaves Pointer to the leaves of the tree.
     * @param config Configuration struct for the merkle tree.
     * @param tree_results Pointer to the tree results (output parameter).
     * @return Error code of type eIcicleError.
     */
    virtual eIcicleError build(const limb_t *leaves, const MerkleTreeConfig& config, limb_t *tree_results /*OUT*/) const = 0;

    /**
     * @brief Pure virtual function to get the path from a specified element index.
     * @param leaves Pointer to the leaves of the tree.
     * @param tree_results Pointer to the tree results.
     * @param element_idx Index of the element for which the path is to be retrieved - counting in elements, not limbs.
     * @param path Pointer to the path output (output parameter). The path is composed of all the input elements.
     * @param config Configuration struct for the merkle tree.
     * @return Error code of type eIcicleError.
     */
    virtual eIcicleError get_path(const limb_t *leaves, const limb_t *tree_results, unsigned int element_idx, limb_t *path /*OUT*/, const MerkleTreeConfig& config) const = 0;

    //

    /**
     * @brief Pure virtual function to verify an element against a Merkle path.
     * @param path Pointer to the Merkle path. The path is composed of all the input elements.
     * @param element_idx Index of the element to verify - counting in elements, not limbs.
     * @param verification_valid Indicates if the verification is valid (output parameter).
     * @param config Configuration struct for the merkle tree.
     * @return Error code of type eIcicleError.
     */
    virtual eIcicleError verify(const limb_t *path, unsigned int element_idx, bool& verification_valid /*OUT*/, const MerkleTreeConfig& config) const = 0;

    // TODO: Temporary for debug

    virtual int get_path_nof_limbs() const = 0;
    virtual eIcicleError print_tree() const = 0;
    virtual eIcicleError print_path(const limb_t *path) const = 0;
    virtual eIcicleError get_node(const limb_t* leaves, int layer_index, int element_index, limb_t* node /*OUT*/) const =0;

    // TODO: Add get_digest() method.
};