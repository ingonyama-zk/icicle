#pragma once

#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "hash.h"
#include "tasks_manager.h"
#include "icicle/utils/utils.h"

#include <cstdint>
#include <functional>


namespace icicle {

/**
 * @brief Configuration struct for the merkle tree.
 */
struct MerkleTreeConfig {
    bool are_leaves_on_device = false;  ///< True if leaves are on device, false if on host. Default is false.
    bool are_tree_results_on_device = false; ///< True if tree results are on device, false if on host. Default is false.
    bool is_path_on_device = false;     ///< True if path is on device, false if on host. Default is false.
    bool is_async = false;              ///< True to run the tree builder asynchronously, false to run it synchronously. Default is false.
    ConfigExtension* ext = nullptr; ///< Backend-specific extensions.
};

/**
 * @brief Class representing a Merkle tree.
 */
class MerkleTree {
 public:
    /**
     * @brief Constructor for the MerkleTree class.
     * @param nof_layers Number of layers in the tree.
     * @param layer_hashes Array of pointers to hashes for each layer.
     * @param output_store_min_layer Minimum layer index to store in the output.
     */
    MerkleTree(
      const unsigned int nof_layers,
      const Hash         *layer_hashes,
      const unsigned int leaf_element_size_in_limbs,
      const unsigned int output_store_min_layer = 0);

    /**
     * @brief Build the Merkle tree from the leaves.
     * @param leaves Pointer to the leaves of the tree.
     * @param config Configuration struct for the merkle tree.
     * @param secondery_leaves Pointer to the secondary_leaves in case the tree recieve inputs from secondary stream.
     * @return Error code of type eIcicleError.
     */
    virtual eIcicleError build(const limb_t *leaves, const MerkleTreeConfig& config, const limb_t *secondery_leaves = nullptr);

    /**
     * @brief Build the Merkle tree from the leaves.
     * @param leaves Pointer to the leaves of the tree.
    */
    virtual eIcicleError get_root(const limb_t* &root) const;


    /**
     * @brief Allocate the path.
     * @param path Pointer to the path pointer.
     * @param nof_limbs Size of the path allocated.
     * @param config Configuration struct for the merkle tree.
     */
    eIcicleError allocate_path(limb_t* &path, unsigned& nof_limbs);

    /**
     * @brief Get the path from a specified element index.
     * @param leaves Pointer to the leaves of the tree.
     * @param tree_results Pointer to the tree results.
     * @param element_idx Index of the element for which the path is to be retrieved - counting in elements, not limbs.
     * @param path Pointer to the path output (output parameter). The path is composed of all the input elements.
     * @param config Configuration struct for the merkle tree.
     * @return Error code of type eIcicleError.
     */
    virtual eIcicleError get_path(const limb_t *leaves, uint64_t element_idx, limb_t *path /*OUT*/, const MerkleTreeConfig& config) const;

    /**
     * @brief Pure virtual function to verify an element against a Merkle path.
     * @param path Pointer to the Merkle path. The path is composed of all the input elements.
     * @param element_idx Index of the element to verify - counting in elements, not limbs.
     * @param verification_valid Indicates if the verification is valid (output parameter).
     * @param config Configuration struct for the merkle tree.
     * @return Error code of type eIcicleError.
     */
    eIcicleError verify(const limb_t *path, unsigned int element_idx, bool& verification_valid /*OUT*/, const MerkleTreeConfig& config);

    // Debug functions
    eIcicleError print_tree() const;
    eIcicleError print_path(const limb_t *path) const;
    eIcicleError get_hash_result(int layer_index, int hash_index, const limb_t* &hash_result /*OUT*/) const;

  private:
    // data base for each layer at the merkle tree
    struct LayerDB {
      const Hash*           m_hash;                               // the hash function
      int64_t               m_secondary_input_offset;             // offset at the secondery input array where this layer start process
      int64_t               m_nof_hashes;                         // number of hash functions.
      std::vector <limb_t>  m_results;                            // vector of hash results. This vector might not be fully allocated if layer is not in range m_output_store_min/max_layer
    };

    // 
    class SegmentDB {
      public:
      SegmentDB(int nof_limbs_to_allocate);
      ~SegmentDB();

      // members
      limb_t*   m_inputs_limbs;
      int       m_nof_inputs_ready;
    };

    class HashTask : public TaskBase {
    public:
      // Constructor
      HashTask() : TaskBase() {}

      // The worker execute this function based on the member operands
      virtual void execute() {
        m_hash->run_multiple_hash(m_input, m_output, m_nof_hashes, *m_hash_config, m_secondary_input);
      } 

      uint64_t      m_nof_hashes;
      const Hash*   m_hash;
      const limb_t* m_input;
      limb_t*       m_output;
      const limb_t* m_secondary_input;
      HashConfig*   m_hash_config;

      // used by the manager
      uint          m_layer_idx;
      int64_t       m_segment_idx;
      uint64_t      m_next_segment_idx;
    };

    // private memebrs
    bool                                          m_tree_already_built;       // inidicated if build function already called
    const unsigned int                            m_nof_layers;               // number of layers in the tree. Each layer is a hash function and results 
    const unsigned int                            m_leaf_element_size_in_limbs;       // the size of one ellement in limbs
    const unsigned int                            m_output_store_min_layer;   // store data starting from this layer
    unsigned int                                  m_path_size_in_limbs;       // number of limbs appear in path

    // Data base per layer
    std::vector < LayerDB >                       m_layers;
    
    // Map from in hash-segment-id to the number of limbs available for process
    // If this segment is not stored in the tree then SegmentDB also contains the input limbs for that segment 
    std::unordered_map< uint64_t, SegmentDB*>  m_map_segment_id_2_inputs;
    
    // config when calling hash function
    HashConfig                                    m_hash_config;

    // get the number of workers to launch at the task manager
    int get_nof_workers(const MerkleTreeConfig& config);

    // Allocate tree results memory and update m_layers with the required data
    void init_layers_db();

    void print_limbs(const limb_t *limbs, const uint nof_limbs) const;

    void build_hash_config_from_merkle_config(const MerkleTreeConfig& merkle_config);

    // build task and dispatch it to task manager
    void dispatch_task(HashTask* task, int cur_layer_idx, const uint64_t cur_segment_idx, const limb_t* input_limbs, const limb_t* secondary_leaves);

};






}; // namespace icicle