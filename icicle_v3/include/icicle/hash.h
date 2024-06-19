#pragma once

#include "errors.h"
#include "runtime.h"
#include "icicle/utils/utils.h"

#include <cstdint>
#include <functional>


typedef uint32_t limb_t;
namespace icicle {
/*************************** Frontend & Backend shared APIs ***************************/
/**
 * @brief Abstract class representing a hash function.
 */
class Hash {
 public:
    const int element_nof_limbs;    ///< Number of limbs in a single hash element.
    const int input_nof_elements;   ///< Number of input elements.
    const int output_nof_elements;  ///< Number of output elements.

    /**
     * @brief Constructor for the Hash class.
     * @param element_nof_limbs Number of limbs in a single hash element.
     * @param input_nof_elements Number of input elements.
     * @param output_nof_elements Number of output elements.
     */
    Hash(int element_nof_limbs, int input_nof_elements, int output_nof_elements)
        : element_nof_limbs{element_nof_limbs}, input_nof_elements{input_nof_elements}, output_nof_elements{output_nof_elements} {}

    /**
     * @brief Pure virtual function to hash multiple input elements.
     * @param input_limbs Pointer to the input limbs.
     * @param output_limbs Pointer to the output limbs.
     * @param batch Number of elements to hash.
     * @return Error code of type eIcicleError.
     */
    virtual eIcicleError hash_many(const limb_t *input_limbs, limb_t *output_limbs, unsigned int batch) const = 0;
};

/**
 * @brief Configuration struct for the tree builder.
 */
struct TreeBuilderConfig {
    bool are_inputs_on_device = false;  ///< True if inputs are on device, false if on host. Default is false.
    bool are_outputs_on_device = false; ///< True if outputs are on device, false if on host. Default is false.
    bool is_async = false;              ///< True to run the tree builder asynchronously, false to run it synchronously. Default is false.
};

/**
 * @brief Class representing a Merkle tree.
 */
class MerkleTree {
 public:
    const unsigned int nof_layers;             ///< Number of layers in the tree.
    const Hash **layer_hashes;                  ///< Array of hashes for each layer (first for leaves, last for root).
    const unsigned int output_store_min_layer; ///< Minimum layer index to store in the output (start index = 0, max_index = nof_layers - 1).
    const unsigned int output_store_max_layer; ///< Maximum layer index to store in the output (start index = 0, max_index = nof_layers - 1).
    const TreeBuilderConfig tree_config;       ///< Configuration for the tree builder.

    limb_t **outputs; /**< Array of pointers to the output of each layer. The size of each array is derived from the tree structure implied by layer_hashes. */

    /**
     * @brief Constructor for the MerkleTree class.
     * @param nof_layers Number of layers in the tree.
     * @param layer_hashes Array of pointers to hashes for each layer.
     * @param output_store_min_layer Minimum layer index to store in the output.
     * @param output_store_max_layer Maximum layer index to store in the output.
     * @param tree_config Configuration for the tree builder.
     */
    MerkleTree(unsigned int nof_layers, const Hash **layer_hashes,
               unsigned int output_store_min_layer, unsigned int output_store_max_layer,
               TreeBuilderConfig tree_config)
        : nof_layers{nof_layers}, layer_hashes{layer_hashes}, 
          output_store_min_layer{output_store_min_layer}, output_store_max_layer{output_store_max_layer}, 
          tree_config{tree_config} {}

    /**
     * @brief Pure virtual function to build the Merkle tree from the leaves.
     * @param leaves Pointer to the leaves of the tree.
     * @return Error code of type eIcicleError.
     */
    virtual eIcicleError build(const limb_t *leaves) const = 0;

    /**
     * @brief Pure virtual function to get the path from a specified element index.
     * @param leaves Pointer to the leaves of the tree.
     * @param element_index Index of the element for which the path is to be retrieved.
     * @param path Pointer to the path output (output parameter).
     * @return Error code of type eIcicleError.
     */
    virtual eIcicleError get_path(const limb_t *leaves, unsigned int element_index, limb_t *path /*OUT*/) const = 0;

    /**
     * @brief Pure virtual function to verify an element against a Merkle path.
     * @param path Pointer to the Merkle path.
     * @param element_idx Index of the element to verify.
     * @param element Pointer to the element to verify.
     * @return Error code of type eIcicleError.
     */
    virtual eIcicleError verify(const limb_t *path, unsigned int element_idx, const limb_t *element, bool& verification_valid /*OUT*/) const = 0;

    // TODO: Temporary for debug

    virtual int get_path_nof_limbs() const = 0;
    virtual eIcicleError print_tree() const = 0;
    virtual eIcicleError print_path(const limb_t *path) const = 0;
    virtual eIcicleError get_node(const limb_t* leaves, int layer_index, int element_index, limb_t* node /*OUT*/) const =0;

    // TODO: Add get_digest() method.
};
/* FRONTEND */
eIcicleError merkle_tree(MerkleTree** merkle_tree, int nof_layers, const Hash **layer_hashes,
               unsigned int output_store_min_layer, unsigned int output_store_max_layer,
               TreeBuilderConfig tree_config);


eIcicleError poseidon(Hash** hash, int element_nof_limbs, int input_nof_elements, int output_nof_elements);

/* BACKEND function registration */
/* Merkle Tree */
using MerkleTreeImpl = std::function<eIcicleError(const Device& device, MerkleTree** merkle_tree, int nof_layers, const Hash **layer_hashes,
               unsigned int output_store_min_layer, unsigned int output_store_max_layer,
               TreeBuilderConfig tree_config)>;

  void register_merkle_tree(const std::string& deviceType, MerkleTreeImpl impl);

#define REGISTER_MERKLE_TREE_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_merkle_tree) = []() -> bool {                                                                  \
      register_merkle_tree(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }


/* POSEIDON HASH */
using PoseidonImpl = std::function<eIcicleError(const Device& device, Hash** hash, int element_nof_limbs, 
                int input_nof_elements, int output_nof_elements)>;


  void register_poseidon(const std::string& deviceType, PoseidonImpl impl);

#define REGISTER_POSEIDON_BACKEND(DEVICE_TYPE, FUNC)                                                                 \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_poseidon) = []() -> bool {                                                                  \
      register_poseidon(DEVICE_TYPE, FUNC);                                                                          \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

}