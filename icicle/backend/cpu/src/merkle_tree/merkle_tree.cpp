#include "merkle_tree.h"
#include <vector>
#include <unordered_map>
#include <iostream>
#define NOF_OPERATIONS_PER_TASK 16
#define CONFIG_NOF_THREADS_KEY  "n_threads" // TODO move it similar to n threads in msm

// class MerkleTree ==//////////////////
// Constructor
MerkleTree::MerkleTree(
  const unsigned int nof_layers, 
  const Hash         *layer_hashes,
  const unsigned int leaf_element_size_in_limbs,
  const unsigned int output_store_min_layer)
  : m_tree_already_built(false),
    m_nof_layers(nof_layers),
    m_layers(nof_layers),                         // vectors of LayerDB per tree layer
    m_leaf_element_size_in_limbs(leaf_element_size_in_limbs),       
    m_output_store_min_layer(output_store_min_layer){

  ICICLE_ASSERT(output_store_min_layer < nof_layers) << "output_store_min_layer must be smaller than nof_layers. At least the root should be saved on tree.";

  // update layers data base with the hashes
  m_path_size_in_limbs = layer_hashes[nof_layers - 1].m_total_output_limbs; // include the root at the path
  for (int layer_idx = nof_layers-1; layer_idx >= 0; --layer_idx) {
    ICICLE_ASSERT(layer_idx == nof_layers-1 || 
                  layer_hashes[layer_idx+1].m_total_input_limbs % layer_hashes[layer_idx].m_total_output_limbs ==0) 
                  << "Each layer output size must divide the above layer input size. Otherwise its not a tree.\n"
                  << "Layer " << layer_idx << " input size = " << layer_hashes[layer_idx+1].m_total_input_limbs << "\n"
                  << "Layer " << layer_idx+1 << " output size = " << layer_hashes[layer_idx].m_total_output_limbs << "\n";

    // initialize m_layers with hashes
    m_layers[layer_idx].m_hash = &layer_hashes[layer_idx];
    
    // Caculate m_path_size_in_limbs
    m_path_size_in_limbs += layer_hashes[layer_idx].m_total_input_limbs;
  }
}

// build the tree and update m_layers with the tree results
eIcicleError MerkleTree::build(const limb_t *leaves, const MerkleTreeConfig& config, const limb_t *secondary_leaves) {
  TasksManager<HashTask> task_manager(get_nof_workers(config));             // Run workers.
  ICICLE_ASSERT(!m_tree_already_built) << "Tree cannot be built more than one time";
  m_tree_already_built = true;
  uint64_t l0_segment_idx = 0;
  init_layers_db();
  const uint64_t nof_segments_at_l0 = (m_layers[0].m_nof_hashes+NOF_OPERATIONS_PER_TASK-1)/NOF_OPERATIONS_PER_TASK;
  
  build_hash_config_from_merkle_config(config);

  // run until the root is processed  
  while (1) {
    HashTask* task = (l0_segment_idx < nof_segments_at_l0) ?        // If there are tasks from layer 0 to send
        task_manager.get_idle_or_completed_task() :                 // get any task slot to assign
        task_manager.get_completed_task();                          // else, only completed tasks are interesting.
    

    // hanlde completed task
    if (task->is_completed()) {
      if (task->m_layer_idx == m_nof_layers - 1) {  // Root processed
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

      // update m_map_segment_id_2_inputs with the limbs that are ready for process
      auto cur_segment_iter = m_map_segment_id_2_inputs.find(cur_segment_id);
      cur_segment_iter->second->m_nof_inputs_ready += m_layers[completed_layer_idx].m_hash->m_total_output_limbs * task->m_nof_hashes;

      // check if cur segment is ready to be executed
      const Hash* cur_hash = m_layers[cur_layer_idx].m_hash;
      const uint64_t nof_hashes_in_seg = std::min(m_layers[cur_layer_idx].m_nof_hashes - cur_segment_idx * NOF_OPERATIONS_PER_TASK, uint64_t(NOF_OPERATIONS_PER_TASK));
      ICICLE_ASSERT(nof_hashes_in_seg > 0) << "Edge case negative number of hashes";
      if (cur_segment_iter->second->m_nof_inputs_ready >= cur_hash->m_total_input_limbs * nof_hashes_in_seg) {
        const limb_t* task_input = (completed_layer_idx < m_output_store_min_layer) ? cur_segment_iter->second->m_inputs_limbs : &(m_layers[completed_layer_idx].m_results[0]);
        dispatch_task(task, cur_layer_idx, cur_segment_idx, task_input, secondary_leaves);
        continue;
      }
    }
    
    if (l0_segment_idx < nof_segments_at_l0) {
      // send a task from layer 0
      dispatch_task(task, 0, l0_segment_idx, leaves, secondary_leaves);
      l0_segment_idx++;
      continue;
    }
    task->set_idle();
  }
}

eIcicleError MerkleTree::get_root(const limb_t* &root) const {
  ICICLE_ASSERT(m_tree_already_built) << "Tree cannot provide root before built";
  root = &(m_layers[m_nof_layers-1].m_results[0]);
  return eIcicleError::SUCCESS;
}

// allocate tree results array based on the size of the tree
eIcicleError MerkleTree::allocate_path(limb_t* &path, unsigned int& path_size) {
  // Path size allready calculates at the constructor.
  path_size = m_path_size_in_limbs;
  path = new limb_t[m_path_size_in_limbs];
  return path ? eIcicleError::SUCCESS : eIcicleError::ALLOCATION_FAILED;
}

eIcicleError MerkleTree::get_path(const limb_t *leaves, uint64_t element_idx, limb_t *path /*OUT*/, const MerkleTreeConfig& config) const {
  ICICLE_ASSERT(m_tree_already_built) << "Tree cannot get path before built";
  const uint64_t element_offset_in_limbs = element_idx * m_leaf_element_size_in_limbs;
  const int l0_total_input_limb =  m_layers[0].m_hash->m_total_input_limbs;
  ICICLE_ASSERT(element_idx < m_layers[0].m_nof_hashes * l0_total_input_limb) << "Element index out of range. Should be smaller than " << m_layers[0].m_nof_hashes * l0_total_input_limb / m_leaf_element_size_in_limbs;
  const uint64_t leaves_offset = (element_offset_in_limbs / l0_total_input_limb) * l0_total_input_limb;

  // if all tree layers stored then copy the leave to the path
  if (m_output_store_min_layer == 0) {
    // round element_offset_in_limbs to the start of l0 hash inputs 
    std::memcpy(path, &leaves[leaves_offset], l0_total_input_limb*sizeof(limb_t));
    path += l0_total_input_limb;
  }
  else {  // not all leaves store, 
    // Define a new tree tree to retrieve the forgotten hash results
    MerkleTree sub_tree (m_output_store_min_layer+1, m_layers[0].m_hash, m_leaf_element_size_in_limbs, 0);
    
    // build the sub tree
    const uint64_t sub_tree_leaves_size = l0_total_input_limb * m_layers[0].m_nof_hashes / m_layers[m_output_store_min_layer].m_nof_hashes;
    const uint64_t sub_tree_leaves_offset = element_offset_in_limbs / sub_tree_leaves_size * sub_tree_leaves_size;
    const limb_t *sub_tree_leaves = &(leaves[sub_tree_leaves_offset]);
    sub_tree.build(sub_tree_leaves, config);
    
    // retrive from the sub tree the path and incremenet path
    const uint64_t sub_tree_nof_elements = sub_tree_leaves_size / m_leaf_element_size_in_limbs;
    const uint64_t sub_tree_element_idx  = element_idx % sub_tree_nof_elements;
    sub_tree.get_path(sub_tree_leaves, sub_tree_element_idx, path, config); 
    path += sub_tree.m_path_size_in_limbs - 1;
  }
  
  // Copy the rest of the path from the stored hash results
  const u_int64_t total_input_elements = m_layers[0].m_nof_hashes * m_layers[0].m_hash->m_total_input_limbs / m_leaf_element_size_in_limbs;
  for (int layer_idx = m_output_store_min_layer; layer_idx < m_nof_layers-1; layer_idx++) {
    uint64_t copy_range_size  =  m_layers[layer_idx+1].m_hash->m_total_input_limbs;  
    uint64_t element_start    =  leaves_offset * m_layers[layer_idx].m_nof_hashes * m_layers[layer_idx].m_hash->m_total_output_limbs / total_input_elements;
    uint64_t copy_range_start =  (element_start / copy_range_size) * copy_range_size;
    auto& cur_layer_result =  m_layers[layer_idx].m_results;

    std::memcpy(path, &(cur_layer_result[copy_range_start]), copy_range_size*sizeof(limb_t));
    path+= copy_range_size;
  }

  // Copy the root.
  const auto& root_layer = m_layers[m_nof_layers-1];
  std::memcpy(path, root_layer.m_results.data(), root_layer.m_hash->m_total_output_limbs*sizeof(limb_t));
  return eIcicleError::SUCCESS; 
}

eIcicleError MerkleTree::verify(const limb_t *path, unsigned int element_idx, bool& verification_valid /*OUT*/, const MerkleTreeConfig& config) {
  build_hash_config_from_merkle_config(config);
  uint64_t element_limb_start = element_idx * m_leaf_element_size_in_limbs;
  for (int layer_idx = 0; layer_idx < m_nof_layers; layer_idx++) {
    const int hash_output_size = m_layers[layer_idx].m_hash->m_total_output_limbs;
    const int hash_input_size = m_layers[layer_idx].m_hash->m_total_input_limbs;
    const int next_hash_input_size = (layer_idx == m_nof_layers - 1) ? hash_output_size : m_layers[layer_idx+1].m_hash->m_total_input_limbs;
    // run the hash on the first limbs of path
    std::vector <limb_t> hash_results(hash_output_size);
    m_layers[layer_idx].m_hash->run_single_hash(path, hash_results.data(), m_hash_config);

    // incrememnt path to the next section of hash inputs
    path += hash_input_size;

    // compare hash results to the element in the path
    element_limb_start = (element_limb_start / hash_input_size) * hash_output_size;
    const int element_limb_offset_in_path = element_limb_start % next_hash_input_size;
    if (std::memcmp(hash_results.data(), path+element_limb_offset_in_path, hash_output_size*sizeof(limb_t))) {
      verification_valid = false;
      return eIcicleError::SUCCESS;
    }
  }
  verification_valid = true;
  return eIcicleError::SUCCESS; 
}
// debug functions //////////////////
eIcicleError MerkleTree::print_tree() const {
  ICICLE_ASSERT(m_tree_already_built) << "Tree cannot print before built";
  for (int layer_idx = m_output_store_min_layer; layer_idx < m_nof_layers; ++layer_idx) {
    std::cout << "Layer " << layer_idx << ":" << std::endl;
    for (const limb_t& limb : m_layers[layer_idx].m_results) {
      std::cout << limb << ", ";
    }
    std::cout << std::endl;
  }
  return eIcicleError::SUCCESS;
}

eIcicleError MerkleTree::print_path(const limb_t *path) const {
  std::cout << "Leaves:" << std::endl;
  print_limbs(path, m_layers[0].m_hash->m_total_input_limbs);
  path += m_layers[0].m_hash->m_total_input_limbs;

  for (uint layer_idx = 0; layer_idx < m_nof_layers-1; ++layer_idx) {
    std::cout << "Layer " << layer_idx << " outputs:" << std::endl;
    print_limbs(path, m_layers[layer_idx+1].m_hash->m_total_input_limbs);
    path += m_layers[layer_idx+1].m_hash->m_total_input_limbs;
  }
  std::cout << "Layer " << m_nof_layers-1 << "(Root):" << std::endl;
  print_limbs(path, m_layers[m_nof_layers-1].m_hash->m_total_output_limbs);
  return eIcicleError::SUCCESS;
}

eIcicleError MerkleTree::get_hash_result(int layer_index, int hash_index, const limb_t* &node /*OUT*/) const {
  ICICLE_ASSERT(m_tree_already_built) << "Tree cannot get hash reult before built";
  ICICLE_ASSERT(layer_index >= m_output_store_min_layer) << "Layer not saved";
  auto& layer = m_layers[layer_index];
  uint result_offset = hash_index * layer.m_hash->m_total_output_limbs;
  node = &(layer.m_results[result_offset]);
  return eIcicleError::SUCCESS;
}

// Private //////////////////

// SegmentDB //////////////////
MerkleTree::SegmentDB::SegmentDB(int nof_limbs_to_allocate) : m_nof_inputs_ready(0) {
  m_inputs_limbs = nof_limbs_to_allocate ? new limb_t[nof_limbs_to_allocate] : nullptr;
}
MerkleTree::SegmentDB::~SegmentDB() {
  if (m_inputs_limbs) {
    delete[] m_inputs_limbs;
  }
}

/////////////////// private functions
int MerkleTree::get_nof_workers(const MerkleTreeConfig& config) {
  if (config.ext && config.ext->has(CONFIG_NOF_THREADS_KEY)) { return config.ext->get<int>(CONFIG_NOF_THREADS_KEY); }

  int hw_threads = std::thread::hardware_concurrency();
  return ((hw_threads > 1) ? hw_threads - 1 : 1); // reduce 1 for the main
}



// allocate tree results array based on the size of the tree
void MerkleTree::init_layers_db() {
  uint64_t nof_hashes = 1;
  // run over all hashes from top layer until bottom layer
  for (int layer_idx = m_nof_layers-1; layer_idx >= 0; --layer_idx) {
    auto&  cur_layer = m_layers[layer_idx];
    cur_layer.m_nof_hashes = nof_hashes;

    // if the layer is in range then allocate the according to the number of hashes in that layer.
    if (m_output_store_min_layer <= layer_idx) {
      const uint64_t nof_limbs_to_allocate = nof_hashes * cur_layer.m_hash->m_total_output_limbs;
      cur_layer.m_results.reserve(nof_limbs_to_allocate);
      cur_layer.m_results.resize(nof_limbs_to_allocate);
    }

    // update nof_hashes to the next layer (below)
    const u_int64_t next_layer_total_output_limbs = (layer_idx > 0) ? m_layers[layer_idx-1].m_hash->m_total_output_limbs : 0;
    nof_hashes = nof_hashes * cur_layer.m_hash->m_total_input_limbs / next_layer_total_output_limbs;
  }

  // run over all layers and update m_secondary_input_offset
  uint64_t secondary_input_offset = 0;
  for (auto layer : m_layers) {
    layer.m_secondary_input_offset = secondary_input_offset;
    secondary_input_offset += layer.m_nof_hashes * layer.m_hash->m_total_secondary_input_limbs;
  }
}

void MerkleTree::print_limbs(const limb_t *limbs, const uint nof_limbs) const {
  for (uint limb_idx = 0; limb_idx < nof_limbs; ++limb_idx) {
    std::cout << *limbs << ", ";
    limbs ++;
  }
  std::cout << std::endl;
}

void MerkleTree::build_hash_config_from_merkle_config(const MerkleTreeConfig& merkle_config) {
  m_hash_config.are_inputs_on_device = merkle_config.are_leaves_on_device;
  m_hash_config.are_outputs_on_device = merkle_config.are_tree_results_on_device;
  m_hash_config.is_async = merkle_config.is_async;

}

void MerkleTree::dispatch_task(HashTask* task, const int cur_layer_idx, const uint64_t cur_segment_idx, const limb_t* input_limbs, const limb_t* secondary_leaves) {
  // Calculate Next-Segment-ID. The current task generates inputs for Next-Segment
  const Hash*    cur_hash = m_layers[cur_layer_idx].m_hash;
  const uint64_t next_layer_idx   = cur_layer_idx+1;

  // Set HashTask input
  task->m_hash = cur_hash;
  const uint64_t input_offset = (cur_layer_idx != 0) && (cur_layer_idx-1 < m_output_store_min_layer) ? 0 : 
                                  cur_segment_idx*NOF_OPERATIONS_PER_TASK*cur_hash->m_total_input_limbs;
  task->m_input = &(input_limbs[input_offset]);

  // Set task secondary input
  const uint64_t task_secondary_input_offset = m_layers[cur_layer_idx].m_secondary_input_offset + cur_segment_idx*NOF_OPERATIONS_PER_TASK*cur_hash->m_total_secondary_input_limbs;
  task->m_secondary_input = &(secondary_leaves[task_secondary_input_offset]);

  task->m_hash = cur_hash;
  task->m_hash_config= &m_hash_config;
  task->m_layer_idx = cur_layer_idx;
  task->m_segment_idx = cur_segment_idx;

  // If this is the last layer
  if (next_layer_idx == m_nof_layers) {
    task->m_nof_hashes = 1;
    task->m_output = m_layers[cur_layer_idx].m_results.data();
    task->dispatch();
    return;
  }

  // This is not the last layer
  const uint64_t next_input_limbs = m_layers[next_layer_idx].m_hash->m_total_input_limbs;
  const uint64_t next_segment_idx = cur_segment_idx * cur_hash->m_total_output_limbs / next_input_limbs;
  const uint64_t next_segment_id  = next_segment_idx ^ (next_layer_idx << 56);

  // If next_segment does not appear m_map_segment_id_2_inputs, add it
  auto next_segment_it = m_map_segment_id_2_inputs.find(next_segment_id);
  if (next_segment_it == m_map_segment_id_2_inputs.end()) {
    const int nof_limbs_to_allocate = (cur_layer_idx < m_output_store_min_layer) ? NOF_OPERATIONS_PER_TASK*next_input_limbs : 0;
    const auto result = m_map_segment_id_2_inputs.emplace(next_segment_id, new SegmentDB(nof_limbs_to_allocate));
    next_segment_it = result.first;
  }

  // calc task_output
  const uint64_t task_output_offset = cur_segment_idx*NOF_OPERATIONS_PER_TASK*cur_hash->m_total_output_limbs;
  task->m_output = (cur_layer_idx < m_output_store_min_layer) ? 
      &(next_segment_it->second->m_inputs_limbs[task_output_offset % (NOF_OPERATIONS_PER_TASK*next_input_limbs)]) :
      &(m_layers[cur_layer_idx].m_results[task_output_offset]);

  // calc task_nof_hashes
  const uint64_t nof_hashes_left = m_layers[cur_layer_idx].m_nof_hashes - next_segment_idx*NOF_OPERATIONS_PER_TASK;
  task->m_nof_hashes = std::min(uint64_t(NOF_OPERATIONS_PER_TASK), nof_hashes_left);

  // Set task next segment to handle return data 
  task->m_next_segment_idx = next_segment_idx;

  // dispatch task
  task->dispatch();
}


// TODO
// destructor 
