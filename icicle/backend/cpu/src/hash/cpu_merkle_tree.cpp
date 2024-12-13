#include <algorithm>
#include <iostream>
#include <iomanip>
#include "icicle/backend/merkle/merkle_tree_backend.h"
#include "icicle/errors.h"
#include "icicle/utils/log.h"
#include "icicle/utils/utils.h"
#include "tasks_manager.h"

#define CONFIG_NOF_THREADS_KEY  "n_threads"
#define NOF_OPERATIONS_PER_TASK 16

namespace icicle {

  class CPUMerkleTreeBackend : public MerkleTreeBackend
  {
  public:
    CPUMerkleTreeBackend(
      const std::vector<Hash>& layer_hashes, uint64_t leaf_element_size, uint64_t output_store_min_layer = 0)
        : MerkleTreeBackend(layer_hashes, leaf_element_size, output_store_min_layer), m_tree_already_built(false)
    {
      // update layers data base with the hashes
      const int nof_layers = layer_hashes.size();
      m_layers.resize(nof_layers);
      m_pruned_path_size = 0;
      m_full_path_size = 0;
      uint64_t nof_hashes = 1;
      for (int layer_idx = nof_layers - 1; layer_idx >= 0; --layer_idx) {
        ICICLE_ASSERT(
          layer_idx == nof_layers - 1 ||
          layer_hashes[layer_idx + 1].default_input_chunk_size() % layer_hashes[layer_idx].output_size() == 0)
          << "Each layer output size must divide the next layer input size. Otherwise its not a tree.\n"
          << "Layer " << layer_idx << " input size = " << layer_hashes[layer_idx + 1].default_input_chunk_size() << "\n"
          << "Layer " << layer_idx + 1 << " output size = " << layer_hashes[layer_idx].output_size() << "\n";

        // initialize m_layers with hashes
        auto& cur_layer = m_layers[layer_idx];
        cur_layer.m_hash = layer_hashes[layer_idx];
        cur_layer.m_nof_hashes = nof_hashes;

        if (0 < layer_idx) {
          // update nof_hashes to the next layer (below)
          const uint64_t cur_layer_input_size = layer_hashes[layer_idx].default_input_chunk_size();
          nof_hashes = nof_hashes * cur_layer_input_size / layer_hashes[layer_idx - 1].output_size();

          // Calculate path_size
          m_pruned_path_size += cur_layer_input_size - layer_hashes[layer_idx - 1].output_size();
          m_full_path_size += cur_layer_input_size;
        }
      }
    }

    eIcicleError build(const std::byte* leaves, uint64_t leaves_size, const MerkleTreeConfig& config) override
    {
      TasksManager<HashTask> task_manager(get_nof_workers(config)); // Run workers.
      if (m_tree_already_built) {
        ICICLE_LOG_ERROR << "Tree cannot be built more than one time";
        return eIcicleError::INVALID_ARGUMENT;
      }
      // build a vector with the leaves that needs padding.
      std::vector<std::byte> padded_leaves;
      if (!init_layers_db(config, leaves_size) || !init_padded_leaves(padded_leaves, leaves, leaves_size, config)) {
        return eIcicleError::INVALID_ARGUMENT;
      }
      const int nof_layers = m_layers.size();
      m_tree_already_built = true; // Set the tree status as built
      uint64_t l0_segment_idx = 0; // Index for the segment of hashes from layer 0 to send
      const uint64_t nof_segments_at_l0 =
        ((m_layers[0].m_nof_hashes_2_execute - m_layers[0].m_last_hash_config.batch) / NOF_OPERATIONS_PER_TASK) + 1;
      bool padding_required = !padded_leaves.empty();

      // run until the root is processed
      while (1) {
        HashTask* task = (l0_segment_idx < nof_segments_at_l0) ? // If there are tasks from layer 0 to send
                           task_manager.get_idle_or_completed_task()
                                                               : // get any task slot to assign
                           task_manager.get_completed_task();    // else, only completed tasks are interesting.

        // handle completed task
        if (task->is_completed()) {
          if (task->m_layer_idx == nof_layers - 1) { // Root processed
            // print_tree(leaves, leaves_size);
            return eIcicleError::SUCCESS;
          }
          const uint64_t completed_layer_idx = task->m_layer_idx;
          const uint64_t completed_segment_idx = task->m_segment_idx;

          // delete completed_segment_id from the map
          const uint64_t completed_segment_id = completed_segment_idx ^ (completed_layer_idx << 56);
          auto segment = m_map_segment_id_2_inputs.find(completed_segment_id);
          if (segment != m_map_segment_id_2_inputs.end()) { m_map_segment_id_2_inputs.erase(completed_segment_id); }

          // Calculate Current-Segment-ID. The completed task generated inputs for Current-Segment
          const uint64_t cur_layer_idx = completed_layer_idx + 1;
          const uint64_t cur_segment_idx = task->m_next_segment_idx;
          const uint64_t cur_segment_id = cur_segment_idx ^ (cur_layer_idx << 56);

          // update m_map_segment_id_2_inputs with the data that is ready for process
          auto cur_segment_iter = m_map_segment_id_2_inputs.find(cur_segment_id);
          cur_segment_iter->second->increment_ready(
            m_layers[completed_layer_idx].m_hash.output_size() * task->m_hash_config->batch);

          // check if cur segment is ready to be executed
          const Hash cur_hash = m_layers[cur_layer_idx].m_hash;
          const uint64_t nof_hashes_in_seg = std::min(
            m_layers[cur_layer_idx].m_nof_hashes_2_execute - cur_segment_idx * NOF_OPERATIONS_PER_TASK,
            uint64_t(NOF_OPERATIONS_PER_TASK));
          if (cur_segment_iter->second->is_ready()) {
            const auto task_input = (completed_layer_idx < m_output_store_min_layer)
                                      ? cur_segment_iter->second->m_input_data.get()
                                      : m_layers[completed_layer_idx].m_results.data();
            dispatch_task(task, cur_layer_idx, cur_segment_idx, task_input, cur_layer_idx > m_output_store_min_layer);
            continue;
          }
        }

        // send task from layer 0:
        // If leaves data is available, send a task based on leaves
        if (l0_segment_idx + padding_required < nof_segments_at_l0) {
          dispatch_task(task, 0, l0_segment_idx, leaves, true);
          l0_segment_idx++;
          continue;
        }

        // If padding is required
        if (padding_required) {
          dispatch_task(task, 0, l0_segment_idx, padded_leaves.data(), false);
          padding_required = false;
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

    eIcicleError get_merkle_proof(
      const std::byte* leaves,
      uint64_t leaves_size,
      uint64_t leaf_idx,
      bool is_pruned,
      const MerkleTreeConfig& config,
      MerkleProof& merkle_proof) const override
    {
      if (!m_tree_already_built) {
        ICICLE_LOG_ERROR << "Tree cannot generate a merkle proof before built\n";
        return eIcicleError::INVALID_ARGUMENT;
      }
      const uint64_t element_offset = leaf_idx * m_leaf_element_size;
      const int l0_total_input_size = m_layers[0].m_hash.default_input_chunk_size();
      if (leaf_idx > m_layers[0].m_nof_hashes * l0_total_input_size) {
        ICICLE_LOG_ERROR << "Leaf index (" << leaf_idx << ") out of range. Should be smaller than "
                         << m_layers[0].m_nof_hashes * l0_total_input_size / m_leaf_element_size;
      }

      const auto [root, root_size] = get_merkle_root();
      // location of the leaves to copy to the proof
      uint64_t proof_leaves_offset = (element_offset / l0_total_input_size) * l0_total_input_size;
      // leaf size at the proof
      const auto proof_leaves_size = m_layers[0].m_hash.default_input_chunk_size();
      // calc the amount of leaves to copy to the proof
      uint64_t copy_leaves_size = (proof_leaves_offset + proof_leaves_size <= leaves_size) ? proof_leaves_size
                                                                                           : // all leaves available
                                    std::min(proof_leaves_size, leaves_size - proof_leaves_offset);
      // generate a vector with the proof leaves
      std::vector<std::byte> proof_leaves(proof_leaves_size, std::byte(0));
      std::memcpy(proof_leaves.data(), &leaves[proof_leaves_offset], copy_leaves_size);

      // if PaddingPolicy::LastValue pad the vector with the last value
      if (config.padding_policy == PaddingPolicy::LastValue) {
        const std::byte* last_element = &leaves[leaves_size - m_leaf_element_size];
        while (copy_leaves_size < proof_leaves_size) {
          std::memcpy(proof_leaves.data() + copy_leaves_size, last_element, m_leaf_element_size);
          copy_leaves_size += m_leaf_element_size;
        }
      }

      // allocate merkle_proof memory
      merkle_proof.allocate(is_pruned, leaf_idx, proof_leaves.data(), proof_leaves_size, root, root_size);

      std::byte* path = merkle_proof.allocate_path_and_get_ptr(is_pruned ? m_pruned_path_size : m_full_path_size);

      // if not all results are stored
      if (m_output_store_min_layer != 0) {
        // Define a new tree tree to retrieve the forgotten hash results
        const std::vector<Hash> sub_tree_layer_hashes(
          m_layer_hashes.begin(), m_layer_hashes.begin() + m_output_store_min_layer + 1);
        CPUMerkleTreeBackend sub_tree(sub_tree_layer_hashes, m_leaf_element_size, 0);

        // build the sub tree
        const uint64_t sub_tree_leaves_size =
          l0_total_input_size * m_layers[0].m_nof_hashes / m_layers[m_output_store_min_layer].m_nof_hashes;
        const uint64_t sub_tree_leaves_offset = (element_offset / sub_tree_leaves_size) * sub_tree_leaves_size;
        const std::byte* sub_tree_leaves = &(leaves[sub_tree_leaves_offset]);
        sub_tree.build(sub_tree_leaves, sub_tree_leaves_size, config);

        // retrieve from the sub tree the path and increment path
        const uint64_t sub_tree_proof_leaves_offset = element_offset % sub_tree_leaves_size;
        path = sub_tree.copy_to_path_from_store_min_layer(sub_tree_proof_leaves_offset, is_pruned, path);
      }

      path = copy_to_path_from_store_min_layer(proof_leaves_offset, is_pruned, path);
      // print_proof(merkle_proof);
      return eIcicleError::SUCCESS;
    }

    // Debug functions
    eIcicleError print_tree(const std::byte* leaves, uint64_t leaves_size) const
    {
      std::cout << "Print tree:" << std::endl;
      std::cout << "Leaves: size=" << m_leaf_element_size << std::endl;
      print_bytes(leaves, leaves_size / m_leaf_element_size, m_leaf_element_size);
      for (int layer_idx = m_output_store_min_layer; layer_idx < m_layers.size(); ++layer_idx) {
        std::cout << std::dec << "Layer " << layer_idx << ": " << m_layers[layer_idx].m_hash.default_input_chunk_size()
                  << " -> " << m_layers[layer_idx].m_hash.output_size() << std::endl;
        print_bytes(
          m_layers[layer_idx].m_results.data(), m_layers[layer_idx].m_nof_hashes_2_execute,
          m_layers[layer_idx].m_hash.output_size());
      }
      return eIcicleError::SUCCESS;
    }

    eIcicleError print_proof(const MerkleProof& proof) const
    {
      const auto [leaf_data, leaf_size, leaf_index] = proof.get_leaf();
      std::cout << "Print Proof: is_pruned(" << proof.is_pruned() << "), leaf_idx(" << leaf_index << ")" << std::endl;
      std::cout << "Leaves:" << std::endl;
      print_bytes(leaf_data, leaf_size / m_leaf_element_size, m_leaf_element_size);

      auto [path, path_size] = proof.get_path();
      for (uint layer_idx = 0; layer_idx < m_layers.size() - 1; ++layer_idx) {
        std::cout << "Layer " << layer_idx << " outputs:" << std::endl;
        const uint output_size = m_layers[layer_idx].m_hash.output_size();
        const uint nof_hashes =
          m_layers[layer_idx].m_nof_hashes / m_layers[layer_idx + 1].m_nof_hashes - (proof.is_pruned() ? 1 : 0);

        print_bytes(path, nof_hashes, output_size);
        path += nof_hashes * output_size;
      }
      auto [root, root_size] = proof.get_root();
      std::cout << "Layer " << m_layers.size() - 1 << "(Root):" << std::endl;
      print_bytes(root, 1, root_size);
      return eIcicleError::SUCCESS;
    }

    eIcicleError get_hash_result(int layer_index, int hash_index, const std::byte*& hash_result /*OUT*/) const
    {
      if (!m_tree_already_built) {
        ICICLE_LOG_ERROR << "Tree cannot get hash result before built\n";
        return eIcicleError::INVALID_ARGUMENT;
      }
      if (layer_index < m_output_store_min_layer) {
        ICICLE_LOG_ERROR << "Layer not saved\n";
        return eIcicleError::INVALID_ARGUMENT;
      }
      auto& layer = m_layers[layer_index];
      uint result_offset = hash_index * layer.m_hash.output_size();
      hash_result = &(layer.m_results[result_offset]);
      return eIcicleError::SUCCESS;
    }

  private:
    // data base for each layer at the merkle tree
    struct LayerDB {
      LayerDB() : m_hash(nullptr) {}

      Hash m_hash;                     // the hash function
      uint64_t m_nof_hashes;           // number of hash functions per layer. Maybe can change to m_input_layer_size
      uint64_t m_nof_hashes_2_execute; // number of hash functions that needs to be calculated

      std::vector<std::byte> m_results; // vector of hash results. This vector might not be fully allocated if layer is
                                        // not in range m_output_store_min/max_layer
      HashConfig m_hash_config;         // config when calling a hash function not last in layer
      HashConfig m_last_hash_config;    // config when calling last in layer hash function
    };

    // the result of each hash segments
    class SegmentDB
    {
    public:
      SegmentDB(int input_size, bool allocate_space) : m_nof_inputs_ready(0), m_input_size(input_size)
      {
        m_input_data.reset(allocate_space ? new std::byte[input_size] : nullptr);
      }

      inline void increment_ready(int nof_inputs_ready) { m_nof_inputs_ready += nof_inputs_ready; }

      inline bool is_ready() const { return (m_nof_inputs_ready >= m_input_size); }
      // members
      std::shared_ptr<std::byte[]> m_input_data;
      int m_input_size;
      int m_nof_inputs_ready;
    };

    class HashTask : public TaskBase
    {
    public:
      // Constructor
      HashTask() : TaskBase(), m_hash(nullptr) {}

      // The worker execute this function based on the member operands
      virtual void execute()
      {
        // run the hash runction
        m_hash.hash(m_input, m_hash.default_input_chunk_size(), *m_hash_config, m_output);

        // pad hash result is necessary
        for (int padd_idx = 0; padd_idx < m_padd_output; padd_idx++) {
          const uint64_t padd_offset = m_hash_config->batch * m_hash.output_size();
          memcpy(
            m_output + padd_offset + padd_idx * m_hash.output_size(), // dest: start from padd_offset
            m_output + padd_offset - m_hash.output_size(),            // source: last calculated hash result
            m_hash.output_size());                                    // size: hash result size
        }
      }

      Hash m_hash;
      const std::byte* m_input;
      std::byte* m_output;
      HashConfig* m_hash_config;

      // task definition: set by the manager
      uint m_layer_idx;
      int64_t m_segment_idx;
      uint64_t m_next_segment_idx;
      uint m_padd_output;
    };

    // private members
    bool m_tree_already_built;       // indicates if build function already called
    unsigned int m_pruned_path_size; // pruned proof size in bytes
    unsigned int m_full_path_size;   // non pruned proof size in bytes

    // Data base per layer
    std::vector<LayerDB> m_layers; // data base per layer

    // Map from in hash-segment-id to the data size available for process
    // If this segment is not stored in the tree then SegmentDB also contains the input data for that segment
    std::unordered_map<uint64_t, std::unique_ptr<SegmentDB>> m_map_segment_id_2_inputs;

    // get the number of workers to launch at the task manager
    int get_nof_workers(const MerkleTreeConfig& config)
    {
      if (config.ext && config.ext->has(CONFIG_NOF_THREADS_KEY)) {
        return config.ext->get<int>(CONFIG_NOF_THREADS_KEY);
      }

      int hw_threads = std::thread::hardware_concurrency();
      return ((hw_threads > 1) ? hw_threads - 1 : 1); // reduce 1 for the main
    }

    // Update m_layers when calling to build based on leaves_size and config
    bool init_layers_db(const MerkleTreeConfig& merkle_config, uint64_t leaves_size)
    {
      const uint nof_layers = m_layers.size();

      // Check leaves size
      if (leaves_size > m_layers[0].m_nof_hashes * m_layers[0].m_hash.default_input_chunk_size()) {
        ICICLE_LOG_ERROR << "Leaves size (" << leaves_size << ") exceeds the size of the tree ("
                         << m_layers[0].m_nof_hashes * m_layers[0].m_hash.default_input_chunk_size() << ")\n";
        return false;
      }
      if (
        leaves_size < m_layers[0].m_nof_hashes * m_layers[0].m_hash.default_input_chunk_size() &&
        merkle_config.padding_policy == PaddingPolicy::None) {
        ICICLE_LOG_ERROR << "Leaves size (" << leaves_size << ") is smaller than tree size ("
                         << m_layers[0].m_nof_hashes * m_layers[0].m_hash.default_input_chunk_size()
                         << ") while Padding policy is None\n";
        return false;
      }

      // run over all hashes from bottom layer to root
      for (int layer_idx = 0; layer_idx < nof_layers; ++layer_idx) {
        auto& cur_layer = m_layers[layer_idx];

        // calculate the actual number of hashes to execute based on leaves_size
        const uint64_t hash_input_size = cur_layer.m_hash.default_input_chunk_size();
        const uint64_t hash_output_size = cur_layer.m_hash.output_size();
        // round up the number of hashes and add 1 more for last hash that is fully padded
        const uint64_t nof_hashes_2_execute = (leaves_size + hash_input_size - 1) / hash_input_size + 1;
        // make sure you don't exceed m_nof_hashes
        cur_layer.m_nof_hashes_2_execute = std::min(cur_layer.m_nof_hashes, nof_hashes_2_execute);

        // config when calling not last in layer hash function
        cur_layer.m_hash_config.batch = NOF_OPERATIONS_PER_TASK;
        cur_layer.m_hash_config.is_async = merkle_config.is_async;

        // config when calling last hash function in layer 2-17 hashes
        const uint64_t last_batch_size = cur_layer.m_nof_hashes_2_execute < NOF_OPERATIONS_PER_TASK
                                           ? cur_layer.m_nof_hashes_2_execute
                                           : (cur_layer.m_nof_hashes_2_execute - 2) % NOF_OPERATIONS_PER_TASK + 2;
        cur_layer.m_last_hash_config.batch = std::min(cur_layer.m_nof_hashes_2_execute, last_batch_size);
        cur_layer.m_last_hash_config.is_async = merkle_config.is_async;

        // update leaves_size for the next layer
        leaves_size = (nof_hashes_2_execute - 1) * hash_output_size;
      }

      // allocate the results vectors based on nof_hashes_2_execute of the next layer. part of it might be padded
      for (int layer_idx = 0; layer_idx < nof_layers; ++layer_idx) {
        const uint64_t nof_bytes_to_allocate = (layer_idx == nof_layers - 1)
                                                 ? m_layers[layer_idx].m_hash.output_size()
                                                 : m_layers[layer_idx + 1].m_nof_hashes_2_execute *
                                                     m_layers[layer_idx + 1].m_hash.default_input_chunk_size();
        m_layers[layer_idx].m_results.reserve(nof_bytes_to_allocate);
        m_layers[layer_idx].m_results.resize(nof_bytes_to_allocate);
      }
      return true;
    }

    // If padding is required resize padded_leaves and populate it with the required data.
    bool init_padded_leaves(
      std::vector<std::byte>& padded_leaves,
      const std::byte* leaves,
      uint64_t leaves_size,
      const MerkleTreeConfig& config)
    {
      const uint64_t l0_input_size = m_layers[0].m_hash.default_input_chunk_size();
      if (m_layers[0].m_nof_hashes * l0_input_size == leaves_size) {
        // No padding is required
        return true;
      }

      const uint64_t padded_leaves_size = m_layers[0].m_last_hash_config.batch * l0_input_size;
      padded_leaves.resize(padded_leaves_size, std::byte(0)); // pad the vector with 0

      // The size of the leaves to copy to padded_leaves
      const uint64_t last_segment_tail_size = (leaves_size - 1) % (NOF_OPERATIONS_PER_TASK * l0_input_size) + 1;
      const uint64_t last_segment_offset = leaves_size - last_segment_tail_size;
      memcpy(padded_leaves.data(), leaves + last_segment_offset, last_segment_tail_size);

      // pad with the last element
      if (config.padding_policy == PaddingPolicy::LastValue) {
        if (leaves_size % m_leaf_element_size != 0) {
          ICICLE_LOG_ERROR << "Leaves size (" << leaves_size << ") must divide leaf_element_size ("
                           << m_leaf_element_size << ") when Padding policy is LastValue\n";
          return false;
        }
        // pad with the last element
        for (uint64_t padded_leaves_offset = last_segment_tail_size; padded_leaves_offset < padded_leaves.size();
             padded_leaves_offset += m_leaf_element_size) {
          memcpy(
            padded_leaves.data() + padded_leaves_offset, // dest: pad vector
            leaves + leaves_size - m_leaf_element_size,  // src: last element
            m_leaf_element_size);                        // size 1 element size
        }
      }
      return true;
    }

    // build task and dispatch it to task manager
    void dispatch_task(
      HashTask* task,
      int cur_layer_idx,
      const uint64_t cur_segment_idx,
      const std::byte* input_bytes,
      bool calc_input_offset)
    {
      // Calculate Next-Segment-ID. The current task generates inputs for Next-Segment
      LayerDB& cur_layer = m_layers[cur_layer_idx];
      const uint64_t next_layer_idx = cur_layer_idx + 1;

      // Set HashTask input
      const uint64_t input_offset =
        calc_input_offset ? cur_segment_idx * NOF_OPERATIONS_PER_TASK * cur_layer.m_hash.default_input_chunk_size() : 0;
      task->m_input = &(input_bytes[input_offset]);

      task->m_hash = cur_layer.m_hash;
      task->m_layer_idx = cur_layer_idx;
      task->m_segment_idx = cur_segment_idx;
      task->m_hash_config = &cur_layer.m_last_hash_config;
      task->m_padd_output = 0;

      // If this is the last layer (root)
      if (next_layer_idx == m_layers.size()) {
        task->m_output = cur_layer.m_results.data();
        task->dispatch();
        return;
      }

      // This is not the root layer (root)
      LayerDB& next_layer = m_layers[next_layer_idx];
      const uint64_t next_input_size = next_layer.m_hash.default_input_chunk_size();
      // Ensure next segment does not overflow due to a <NOF_OPERATIONS_PER_TASK+1> sized batch by comparing it to the
      // max possible segment index (And taking the smaller one)
      const uint64_t max_segment_idx =
        (next_layer.m_nof_hashes_2_execute - next_layer.m_last_hash_config.batch) / NOF_OPERATIONS_PER_TASK;
      const uint64_t next_segment_idx =
        std::min(cur_segment_idx * cur_layer.m_hash.output_size() / next_input_size, max_segment_idx);
      const uint64_t next_segment_id = next_segment_idx ^ (next_layer_idx << 56);

      // If next_segment does not appear in m_map_segment_id_2_inputs, then add it
      auto next_segment_it = m_map_segment_id_2_inputs.find(next_segment_id);
      if (next_segment_it == m_map_segment_id_2_inputs.end()) {
        bool is_next_segment_last = next_segment_idx * NOF_OPERATIONS_PER_TASK + next_layer.m_last_hash_config.batch ==
                                    next_layer.m_nof_hashes_2_execute;
        const int next_seg_size_to_allocate =
          is_next_segment_last ? next_layer.m_last_hash_config.batch * next_input_size
                               :                       // last segment - allocate according to batch size
            NOF_OPERATIONS_PER_TASK * next_input_size; // middle segment - allocate max
        const auto result = m_map_segment_id_2_inputs.emplace(
          next_segment_id,
          std::make_unique<SegmentDB>(SegmentDB(next_seg_size_to_allocate, cur_layer_idx < m_output_store_min_layer)));
        next_segment_it = result.first;
      }

      // calc task_output
      const uint64_t task_output_offset = cur_segment_idx * NOF_OPERATIONS_PER_TASK * cur_layer.m_hash.output_size();
      task->m_output =
        (cur_layer_idx < m_output_store_min_layer)
          ? &(next_segment_it->second->m_input_data[task_output_offset % (NOF_OPERATIONS_PER_TASK * next_input_size)])
          :                                           // input in SegmentDB
          &(cur_layer.m_results[task_output_offset]); // next layer result vector

      // If this is the last segment, pad the result
      bool is_cur_segment_last = cur_segment_idx * NOF_OPERATIONS_PER_TASK + cur_layer.m_last_hash_config.batch ==
                                 cur_layer.m_nof_hashes_2_execute;
      if (is_cur_segment_last) {
        // total size of the next layer inputs
        const uint64_t result_total_size = next_layer.m_nof_hashes_2_execute * next_input_size;
        // idx for the last hash at the current segment
        const uint64_t last_result_idx = cur_segment_idx * NOF_OPERATIONS_PER_TASK + cur_layer.m_last_hash_config.batch;
        // location of the hash result at the next layer inputs
        const uint64_t last_result_location = last_result_idx * task->m_hash.output_size();
        const uint64_t padd_size_in_bytes = result_total_size - last_result_location;
        task->m_padd_output = padd_size_in_bytes / task->m_hash.output_size();
        next_segment_it->second->increment_ready(padd_size_in_bytes);
      } else {
        task->m_hash_config = &cur_layer.m_hash_config;
      }

      // Set task next segment to handle return data
      task->m_next_segment_idx = next_segment_idx;

      // send task to the worker for execution
      task->dispatch();
    }

    // restore the proof path from the tree and return the new path pointer
    std::byte*
    copy_to_path_from_store_min_layer(const uint64_t proof_leaves_offset, bool is_pruned, std::byte* path) const
    {
      const uint64_t total_input_size = m_layers[0].m_nof_hashes * m_layers[0].m_hash.default_input_chunk_size();
      for (int layer_idx = m_output_store_min_layer; layer_idx < m_layers.size() - 1; layer_idx++) {
        const auto& cur_layer_result = m_layers[layer_idx].m_results;
        const uint64_t copy_range_size = m_layers[layer_idx + 1].m_hash.default_input_chunk_size();
        const uint64_t one_element_size = m_layers[layer_idx].m_hash.output_size();
        uint64_t element_start =
          proof_leaves_offset * m_layers[layer_idx].m_nof_hashes / total_input_size * one_element_size;

        // if the element exceeds to the padded area, cut it to the padded hash
        if (element_start >= cur_layer_result.size()) {
          element_start = cur_layer_result.size() - copy_range_size + element_start % copy_range_size;
        }
        const uint64_t copy_chunk_start = (element_start / copy_range_size) * copy_range_size;

        for (uint64_t byte_idx = copy_chunk_start; byte_idx < copy_chunk_start + copy_range_size; byte_idx++) {
          if (
            !is_pruned || byte_idx < element_start ||       // copy data before the element
            element_start + one_element_size <= byte_idx) { // copy data after the element
            *path = cur_layer_result[byte_idx];
            path++;
          }
        }
      }
      return path;
    }
  };

  eIcicleError create_merkle_tree_backend(
    const Device& device,
    const std::vector<Hash>& layer_hashes,
    uint64_t leaf_element_size,
    uint64_t output_store_min_layer,
    std::shared_ptr<MerkleTreeBackend>& backend)
  {
    backend = std::make_shared<CPUMerkleTreeBackend>(layer_hashes, leaf_element_size, output_store_min_layer);
    return eIcicleError::SUCCESS;
  }

  REGISTER_MERKLE_TREE_FACTORY_BACKEND("CPU", create_merkle_tree_backend);

} // namespace icicle