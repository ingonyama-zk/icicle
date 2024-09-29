#include <algorithm>
#include <iostream>
#include <iomanip>
#include "icicle/backend/merkle/merkle_tree_backend.h"
#include "icicle/errors.h"
#include "icicle/utils/log.h"
#include "tasks_manager.h"

#define CONFIG_NOF_THREADS_KEY  "n_threads" // TODO move it similar to n threads in msm
#define NOF_OPERATIONS_PER_TASK 16

namespace icicle {

  class CPUMerkleTreeBackend : public MerkleTreeBackend
  {
  public:
    CPUMerkleTreeBackend(
      const std::vector<Hash>& layer_hashes, uint64_t leaf_element_size, uint64_t output_store_min_layer = 0)
        : MerkleTreeBackend(layer_hashes, leaf_element_size, output_store_min_layer),
        m_tree_already_built(false)
    {
      // update layers data base with the hashes
      const int nof_layers = layer_hashes.size();
      m_layers.resize(nof_layers);
      m_prooned_path_size = 0;
      m_full_path_size = 0;
      for (int layer_idx = 0; layer_idx < nof_layers; ++layer_idx) {
        ICICLE_ASSERT(layer_idx == nof_layers-1 || 
                      layer_hashes[layer_idx+1].input_default_chunk_size() % layer_hashes[layer_idx].output_size() ==0) 
                      << "Each layer output size must divide the above layer input size. Otherwise its not a tree.\n"
                      << "Layer " << layer_idx << " input size = " << layer_hashes[layer_idx+1].input_default_chunk_size()<< "\n"
                      << "Layer " << layer_idx+1 << " output size = " << layer_hashes[layer_idx].output_size() << "\n";

        // initialize m_layers with hashes
        m_layers[layer_idx].m_hash = layer_hashes[layer_idx];
        
        // Caculate m_path_size
        if (0 < layer_idx) {
          m_prooned_path_size += layer_hashes[layer_idx].input_default_chunk_size() - layer_hashes[layer_idx-1].output_size();
          m_full_path_size += layer_hashes[layer_idx].input_default_chunk_size();
        }
      }
    }

    // TODO: handle size
    eIcicleError build(const std::byte* leaves, uint64_t leaves_size, const MerkleTreeConfig& config) override
    {
      TasksManager<HashTask> task_manager(get_nof_workers(config));             // Run workers.
      ICICLE_ASSERT(!m_tree_already_built) << "Tree cannot be built more than one time";
      const int nof_layers = m_layers.size();
      m_tree_already_built = true;              // Set the tree status as built
      uint64_t l0_segment_idx = 0;              // Index for the chunk of hashes from layer 0 to send                                
      init_layers_db(config);
      const uint64_t nof_segments_at_l0 = (m_layers[0].m_nof_hashes+NOF_OPERATIONS_PER_TASK-1)/NOF_OPERATIONS_PER_TASK;
      
      // run until the root is processed  
      while (1) {
        HashTask* task = (l0_segment_idx < nof_segments_at_l0) ?        // If there are tasks from layer 0 to send
            task_manager.get_idle_or_completed_task() :                 // get any task slot to assign
            task_manager.get_completed_task();                          // else, only completed tasks are interesting.
        
        // hanlde completed task
        if (task->is_completed()) {
          if (task->m_layer_idx == nof_layers - 1) {  // Root processed
            print_tree(leaves, leaves_size);
            return eIcicleError::SUCCESS;
          }
          const uint64_t completed_layer_idx = task->m_layer_idx;
          const uint64_t completed_segment_idx = task->m_segment_idx;

          // delete completed_segment_id from the map
          const uint64_t completed_segment_id = completed_segment_idx ^ (completed_layer_idx << 56);
          m_map_segment_id_2_inputs.erase(completed_segment_id);

          // Calculate Current-Segment-ID. The completed task generated inputs for Current-Segment
          const uint64_t cur_layer_idx   = completed_layer_idx+1;
          const uint64_t cur_segment_idx = task->m_next_segment_idx;
          const uint64_t cur_segment_id  = cur_segment_idx ^ (cur_layer_idx << 56);

          // update m_map_segment_id_2_inputs with the data that is ready for process
          auto cur_segment_iter = m_map_segment_id_2_inputs.find(cur_segment_id);
          cur_segment_iter->second->m_nof_inputs_ready += m_layers[completed_layer_idx].m_hash.output_size() * NOF_OPERATIONS_PER_TASK;

          // check if cur segment is ready to be executed
          const Hash cur_hash = m_layers[cur_layer_idx].m_hash;
          const uint64_t nof_hashes_in_seg = std::min(m_layers[cur_layer_idx].m_nof_hashes - cur_segment_idx * NOF_OPERATIONS_PER_TASK, uint64_t(NOF_OPERATIONS_PER_TASK));
          ICICLE_ASSERT(nof_hashes_in_seg > 0) << "Edge case negative number of hashes";
          if (cur_segment_iter->second->m_nof_inputs_ready >= cur_hash.input_default_chunk_size() * nof_hashes_in_seg) {
            const std::byte* task_input = (completed_layer_idx < m_output_store_min_layer) ? cur_segment_iter->second->m_input_data : &(m_layers[completed_layer_idx].m_results[0]);
            dispatch_task(task, cur_layer_idx, cur_segment_idx, task_input);
            continue;
          }
        }
        
        if (l0_segment_idx < nof_segments_at_l0) {
          // send a task from layer 0
          dispatch_task(task, 0, l0_segment_idx, leaves);
          l0_segment_idx++;
          continue;
        }
        task->set_idle();
      }
      return eIcicleError::SUCCESS;
    }

    std::pair<const std::byte*, size_t> get_merkle_root() const override
    {
      ICICLE_ASSERT(m_tree_already_built) << "Merkle tree cannot provide root before built";
      return {m_layers.back().m_results.data(), m_layers.back().m_hash.output_size()};
    }

    // TODO: handle leaves_size
    eIcicleError get_merkle_proof(
      const std::byte* leaves,
      uint64_t leaves_size,
      uint64_t leaf_idx,
      bool is_pruned,
      const MerkleTreeConfig& config,
      MerkleProof& merkle_proof) const override
    {
      ICICLE_ASSERT(m_tree_already_built) << "Tree cannot generate a merkle proof before built";
      const uint64_t element_offset = leaf_idx * m_leaf_element_size;
      const int l0_total_input_size =  m_layers[0].m_hash.input_default_chunk_size();
      ICICLE_ASSERT(leaf_idx < m_layers[0].m_nof_hashes * l0_total_input_size) << "Element index out of range. Should be smaller than " << m_layers[0].m_nof_hashes * l0_total_input_size / m_leaf_element_size;
      uint64_t input_chunk_offset = (element_offset / l0_total_input_size) * l0_total_input_size;

      // allocate merkle_proof memory
      const auto [root, root_size] = get_merkle_root();
      const auto input_chunk_size = m_layers[0].m_hash.input_default_chunk_size();
      merkle_proof.allocate(is_pruned, leaf_idx, &leaves[input_chunk_offset], input_chunk_size, root, root_size);   

      std::byte* path = merkle_proof.allocate_path_and_get_ptr(is_pruned ? m_prooned_path_size : m_full_path_size);

      // if not all leaves are stored
      if (m_output_store_min_layer != 0) {
        // Define a new tree tree to retrieve the forgotten hash results
        const std::vector<Hash> sub_tree_layer_hashes(m_layer_hashes.begin(), m_layer_hashes.begin() + m_output_store_min_layer + 1);
        CPUMerkleTreeBackend sub_tree(sub_tree_layer_hashes, m_leaf_element_size, 0);

        // build the sub tree
        const uint64_t sub_tree_leaves_size = l0_total_input_size * m_layers[0].m_nof_hashes / m_layers[m_output_store_min_layer].m_nof_hashes;
        const uint64_t sub_tree_input_chunk_offset = element_offset / sub_tree_leaves_size * sub_tree_leaves_size;
        const std::byte *sub_tree_leaves = &(leaves[sub_tree_input_chunk_offset]);
        sub_tree.build(sub_tree_leaves, sub_tree_leaves_size, config);
        
        // retrive from the sub tree the path and incremenet path
        sub_tree.copy_to_path_from_store_min_layer(input_chunk_offset, is_pruned, path);
      }
      
      copy_to_path_from_store_min_layer(input_chunk_offset, is_pruned, path);
      print_proof(merkle_proof);
      return eIcicleError::SUCCESS; 
    }

    // Debug functions
    eIcicleError print_tree(const std::byte* leaves, uint64_t leaves_size) const {
      ICICLE_ASSERT(m_tree_already_built) << "Tree cannot print before built";
      std::cout << "Print tree:" << std::endl;
      std::cout << "Leaves: size=" << m_leaf_element_size << std::endl;
      print_bytes(leaves, leaves_size/m_leaf_element_size, m_leaf_element_size);
      for (int layer_idx = m_output_store_min_layer; layer_idx < m_layers.size(); ++layer_idx) {
        std::cout << std::dec << "Layer " << layer_idx << ": " << m_layers[layer_idx].m_hash.input_default_chunk_size() << " -> " <<  m_layers[layer_idx].m_hash.output_size() << std::endl;
        print_bytes(m_layers[layer_idx].m_results.data(), m_layers[layer_idx].m_nof_hashes, m_layers[layer_idx].m_hash.output_size());
      }
      return eIcicleError::SUCCESS;
    }

    eIcicleError print_proof(const MerkleProof& proof) const {
      const auto [leaf_data, leaf_size, leaf_index] = proof.get_leaf();
      std::cout << "Print Proof: is_pruned(" << proof.is_pruned() << "), leaf_idx(" << leaf_index << ")" << std::endl;
      std::cout << "Leaves:" << std::endl;
      print_bytes(leaf_data, leaf_size/m_leaf_element_size, m_leaf_element_size);

      auto [path, path_size] = proof.get_path();
      for (uint layer_idx = 0; layer_idx < m_layers.size()-1; ++layer_idx) {
        std::cout << "Layer " << layer_idx << " outputs:" << std::endl;
        const uint output_size = m_layers[layer_idx].m_hash.output_size();
        const uint nof_hashes = m_layers[layer_idx].m_nof_hashes / m_layers[layer_idx+1].m_nof_hashes - 
                                (proof.is_pruned() ? 1 : 0);

        print_bytes(path, nof_hashes, output_size);
        path += nof_hashes*output_size;
      }
      auto [root, root_size] = proof.get_root();
      std::cout << "Layer " << m_layers.size()-1 << "(Root):" << std::endl;
      print_bytes(root, 1, root_size);
      return eIcicleError::SUCCESS;
    }

    eIcicleError get_hash_result(int layer_index, int hash_index, const std::byte* &hash_result /*OUT*/) const {
      ICICLE_ASSERT(m_tree_already_built) << "Tree cannot get hash reult before built";
      ICICLE_ASSERT(layer_index >= m_output_store_min_layer) << "Layer not saved";
      auto& layer = m_layers[layer_index];
      uint result_offset = hash_index * layer.m_hash.output_size();
      hash_result = &(layer.m_results[result_offset]);
      return eIcicleError::SUCCESS;

    }


  private:
    // data base for each layer at the merkle tree
    struct LayerDB {
      Hash                     m_hash;                // the hash function
      int64_t                  m_nof_hashes;          // number of hash functions.
      std::vector <std::byte>  m_results;             // vector of hash results. This vector might not be fully allocated if layer is not in range m_output_store_min/max_layer
      HashConfig               m_hash_config;         // config when calling a hash function not last in layer
      HashConfig               m_last_hash_config;    // config when calling last in layer hash function
      std::vector <std::byte>  m_zero_padded_input;   // contains the last input in case padding is required
      std::vector <std::byte>  m_zero_input;          // zero vector for padded inputs
    };

    // the result of each hash segments
    class SegmentDB {
      public:
      SegmentDB(int size_to_allocate) : m_nof_inputs_ready(0) {
        m_input_data = size_to_allocate ? new std::byte[size_to_allocate] : nullptr;
      }
      ~SegmentDB() {
        if (m_input_data) {
          delete[] m_input_data;
        }
      }

      // members
      std::byte*   m_input_data;
      int          m_nof_inputs_ready;
    };

    class HashTask : public TaskBase {
    public:
      // Constructor
      HashTask() : TaskBase() {}

      // The worker execute this function based on the member operands
      virtual void execute() {
        m_hash.hash(m_input, m_hash.input_default_chunk_size(), *m_hash_config, m_output);
      } 

      Hash                m_hash;
      const std::byte*    m_input;
      std::byte*          m_output;
      HashConfig*         m_hash_config;

      // used by the manager
      uint                m_layer_idx;
      int64_t             m_segment_idx;
      uint64_t            m_next_segment_idx;
    };

    // private memebrs
    bool                                          m_tree_already_built;     // inidicated if build function already called
    unsigned int                                  m_prooned_path_size;      //  prooned proof size in bytes
    unsigned int                                  m_full_path_size;         //  non prooned proof size in bytes

    // Data base per layer
    std::vector < LayerDB >                       m_layers;                 // data base per layer
    
    // Map from in hash-segment-id to the data size available for process
    // If this segment is not stored in the tree then SegmentDB also contains the input data for that segment 
    std::unordered_map< uint64_t, SegmentDB*>  m_map_segment_id_2_inputs;
    
    // get the number of workers to launch at the task manager
    int get_nof_workers(const MerkleTreeConfig& config) {
      if (config.ext && config.ext->has(CONFIG_NOF_THREADS_KEY)) { return config.ext->get<int>(CONFIG_NOF_THREADS_KEY); }

      int hw_threads = std::thread::hardware_concurrency();
      return ((hw_threads > 1) ? hw_threads - 1 : 1); // reduce 1 for the main
    }

    // Allocate tree results memory and update m_layers with the required data
    void init_layers_db(const MerkleTreeConfig& merkle_config) {
      const uint nof_layers = m_layers.size();
      uint64_t nof_hashes = 1;

      // run over all hashes from top layer until bottom layer
      for (int layer_idx = nof_layers-1; layer_idx >= 0; --layer_idx) {
        auto&  cur_layer = m_layers[layer_idx];
        cur_layer.m_nof_hashes = nof_hashes;

        // config when calling a hash function not last in layer
        cur_layer.m_hash_config.batch = NOF_OPERATIONS_PER_TASK;
        cur_layer.m_hash_config.is_async = merkle_config.is_async;

        // config when calling last in layer hash function
        cur_layer.m_last_hash_config.batch = (nof_hashes % NOF_OPERATIONS_PER_TASK) ? (nof_hashes % NOF_OPERATIONS_PER_TASK) : NOF_OPERATIONS_PER_TASK;
        cur_layer.m_last_hash_config.is_async = merkle_config.is_async;

        // if the layer is in range then allocate the according to the number of hashes in that layer.
        if (m_output_store_min_layer <= layer_idx) {
          const uint64_t nof_bytes_to_allocate = nof_hashes * cur_layer.m_hash.input_default_chunk_size();
          cur_layer.m_results.reserve(nof_bytes_to_allocate);
          cur_layer.m_results.resize(nof_bytes_to_allocate);
        }

        // update nof_hashes to the next layer (below)
        const u_int64_t next_layer_total_size = (layer_idx > 0) ? m_layers[layer_idx-1].m_hash.output_size() : 1;
        nof_hashes = nof_hashes * cur_layer.m_hash.input_default_chunk_size() / next_layer_total_size;
      }
    }

    // build task and dispatch it to task manager
    void dispatch_task(HashTask* task, int cur_layer_idx, const uint64_t cur_segment_idx, const std::byte* input_bytes) {
      // Calculate Next-Segment-ID. The current task generates inputs for Next-Segment
      LayerDB& cur_layer = m_layers[cur_layer_idx];
      const uint64_t next_layer_idx   = cur_layer_idx+1;

      // Set HashTask input
      const uint64_t input_offset = (cur_layer_idx != 0) && (cur_layer_idx-1 < m_output_store_min_layer) ? 0 : 
                                      cur_segment_idx*NOF_OPERATIONS_PER_TASK*cur_layer.m_hash.input_default_chunk_size();
      task->m_input = &(input_bytes[input_offset]);

      task->m_hash = cur_layer.m_hash;
      task->m_hash_config = &cur_layer.m_last_hash_config;
      task->m_layer_idx = cur_layer_idx;
      task->m_segment_idx = cur_segment_idx;

      // If this is the last layer
      if (next_layer_idx == m_layers.size()) {
        task->m_output = cur_layer.m_results.data();
        task->dispatch();
        return;
      }

      // This is not the last layer
      const uint64_t next_input_size = m_layers[next_layer_idx].m_hash.input_default_chunk_size();
      const uint64_t next_segment_idx = cur_segment_idx * cur_layer.m_hash.output_size() / next_input_size;
      const uint64_t next_segment_id  = next_segment_idx ^ (next_layer_idx << 56);

      // If next_segment does not appear m_map_segment_id_2_inputs, add it
      auto next_segment_it = m_map_segment_id_2_inputs.find(next_segment_id);
      if (next_segment_it == m_map_segment_id_2_inputs.end()) {
        const int size_to_allocate = (cur_layer_idx < m_output_store_min_layer) ? NOF_OPERATIONS_PER_TASK*next_input_size : 0;
        const auto result = m_map_segment_id_2_inputs.emplace(next_segment_id, new SegmentDB(size_to_allocate));
        next_segment_it = result.first;
      }

      // calc task_output
      const uint64_t task_output_offset = cur_segment_idx*NOF_OPERATIONS_PER_TASK*cur_layer.m_hash.output_size();
      task->m_output = (cur_layer_idx < m_output_store_min_layer) ? 
          &(next_segment_it->second->m_input_data[task_output_offset % (NOF_OPERATIONS_PER_TASK*next_input_size)]) :
          &(cur_layer.m_results[task_output_offset]);

      // If this is not the last hash, update hash config
      if ( (cur_segment_idx+1) * NOF_OPERATIONS_PER_TASK < cur_layer.m_nof_hashes)
        task->m_hash_config = &cur_layer.m_hash_config;

      // Set task next segment to handle return data 
      task->m_next_segment_idx = next_segment_idx;

      // dispatch task
      task->dispatch();      
    }

    // restore the proof path from the tree
    void copy_to_path_from_store_min_layer(uint64_t& input_chunk_offset, bool is_prooned, std::byte* &path) const {
      const u_int64_t total_input_size = m_layers[0].m_nof_hashes * m_layers[0].m_hash.input_default_chunk_size();
      for (int layer_idx = m_output_store_min_layer; layer_idx < m_layers.size()-1; layer_idx++) {
        const uint64_t copy_range_size  =  m_layers[layer_idx+1].m_hash.input_default_chunk_size();  
        const uint64_t one_element_size = m_layers[layer_idx].m_hash.output_size();
        const uint64_t element_start    =  input_chunk_offset * m_layers[layer_idx].m_hash.output_size() / m_layers[layer_idx].m_hash.input_default_chunk_size();
        input_chunk_offset =  (element_start / copy_range_size) * copy_range_size;
        auto& cur_layer_result =  m_layers[layer_idx].m_results;

        for (int byte_idx = input_chunk_offset; byte_idx < input_chunk_offset + copy_range_size; byte_idx++) {
          if (!is_prooned ||
              byte_idx < element_start ||                       // copy data before the element
              element_start + one_element_size <= byte_idx)  {   // copy data after the element
            *path = cur_layer_result[byte_idx];
            path++;
          }
        }
      }
    }

    // Debug
    void print_bytes(const std::byte *data, const uint nof_elements, const uint element_size) const {
      for (uint element_idx = 0; element_idx < nof_elements; ++element_idx) {
        std::cout <<  ", 0x";
        for (int byte_idx = element_size-1; byte_idx >=0; --byte_idx) {
          std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(data[element_idx*element_size + byte_idx]);
        }
      }
      std::cout << std::endl;
    }

  };

  eIcicleError create_merkle_tree_backend(
    const Device& device,
    const std::vector<Hash>& layer_hashes,
    uint64_t leaf_element_size,
    uint64_t output_store_min_layer,
    std::shared_ptr<MerkleTreeBackend>& backend)
  {
    ICICLE_LOG_INFO << "Creating CPU MerkleTreeBackend";
    backend = std::make_shared<CPUMerkleTreeBackend>(layer_hashes, leaf_element_size, output_store_min_layer);
    return eIcicleError::SUCCESS;
  }

  REGISTER_MERKLE_TREE_FACTORY_BACKEND("CPU", create_merkle_tree_backend);

} // namespace icicle