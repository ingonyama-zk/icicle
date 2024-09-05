#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "hash.h"
#include "hashes.h"
#include "icicle/utils/utils.h"
#include "merkle_tree.h"

#include <cassert>
#include <cstdlib>
#include <ctime>

// TODO test with side limbs

void assert_valid_tree(
  const MerkleTree* tree,
  const int nof_layers,
  int nof_inputs,
  const limb_t* inputs,
  const limb_t* side_inputs,
  const Hash** hashes,
  const int output_store_min_layer
)
{ 
  int nof_outputs = nof_inputs * hashes[0]->m_total_output_limbs / hashes[0]->m_total_input_limbs;
  limb_t* layer_in =  new limb_t[nof_inputs];   // Going layer by layer and having the input layer as the largest 
  limb_t* layer_out = new limb_t[nof_outputs];  // ensures  these are the maximum sizes required for inputs and outputs
  // NOTE there is an assumption here that output number is less or equal to input number for all layers

  for (int i = 0; i < nof_inputs; i++) { layer_in[i] = inputs[i]; }
  
  // int nof_hashes = nof_inputs / hashes[0]->m_total_input_limbs;
  int side_inputs_offset = 0;
  for (int i = 0; i < nof_layers; i++)
  {
    nof_outputs = nof_inputs * hashes[i]->m_total_output_limbs / hashes[i]->m_total_input_limbs;
    const int nof_hashes = nof_inputs / hashes[i]->m_total_input_limbs;
    // TODO actual config to allow gpu test as well?
    hashes[i]->run_multiple_hash(layer_in, layer_out, nof_hashes, HashConfig(), &side_inputs[side_inputs_offset]);

    // Assert stored values in the trees (if they exist)    
    if (i >= output_store_min_layer) {
      for (int j = 0; j < nof_hashes; j++) { 
        const limb_t* result = nullptr;
        tree->get_hash_result(i, j, result);
        const int num_outputs = hashes[i]->m_total_output_limbs;
        for (int k = 0; k < num_outputs; k++)
        {
          ICICLE_ASSERT(layer_out[num_outputs*j + k] == *(result+k)) << 
            "\n(" << i << ',' << j << "):\t" << *(result+k) << "\t=\\=\t" << layer_out[num_outputs*j + k] << '\n';
        }
      }
    }

    // Transfer outputs to inputs before moving to the next layer
    for (int i = 0; i < nof_outputs; i++) { layer_in[i] = layer_out[i]; }
    nof_inputs = nof_outputs;
    side_inputs_offset += hashes[i]->m_total_secondary_input_limbs * nof_hashes;
  }

  // Assert final root
  const limb_t* root;
  tree->get_root(root);
  for (int i = 0; i < hashes[nof_layers-1]->m_total_output_limbs; i++) { assert(root[i] == layer_out[i]); }

  delete[] layer_in;
  delete[] layer_out;
}

void assert_valid_path(
  const limb_t* path, int limb_idx, const int nof_layers, const Hash** hashes
)
{

  for (int i = 0; i < nof_layers; i++)
  {
    ICICLE_ASSERT(hashes[i]->m_total_secondary_input_limbs == 0) << "Path generation with side inputs currently cannot be validated\n";
  }
  
  int curr_offset = 0;
  int next_offset = 0;
  for (int i = 0; i < nof_layers - 1; i++)
  {
    limb_t* hash_results = new limb_t[hashes[i]->m_total_output_limbs];
    next_offset += hashes[i]->m_total_input_limbs;
    hashes[i]->run_single_hash(&path[curr_offset], hash_results, HashConfig());
    
    // Compare calculated results with the path
    // Get the appropriate index of the output in the next layer of the path
    limb_idx = (limb_idx / hashes[i]->m_total_input_limbs) * hashes[i]->m_total_output_limbs;
    int res_offset = limb_idx % hashes[i+1]->m_total_input_limbs;
    
    for (int j = 0; j < hashes[i]->m_total_output_limbs; j++) {
      ICICLE_ASSERT(path[next_offset + res_offset + j] ==  hash_results[j]) << "\n(" << i << ',' << j << "):\t" << 
        path[next_offset + res_offset + j] << "\t=\\=\t" << hash_results[j] << '\n'; 
    }
    curr_offset = next_offset;
    delete[] hash_results;
  }

  // Last layer
  limb_t* hash_results = new limb_t[hashes[nof_layers-1]->m_total_output_limbs];
  next_offset += hashes[nof_layers-1]->m_total_input_limbs;
  hashes[nof_layers-1]->run_single_hash(&path[curr_offset], hash_results, HashConfig());
  for (int j = 0; j < hashes[nof_layers-1]->m_total_output_limbs; j++) { 
    ICICLE_ASSERT(path[next_offset + j] ==  hash_results[j]) << 
      "\n(" << nof_layers-1 << ',' << j << "):\t" << path[next_offset+ j] << "\t=\\=\t" << hash_results[j] << '\n'; 
  }
  delete[] hash_results;
}

int random(int low, int high)
{
  return low + (rand() % (high - low));
}

eIcicleError tree_test(bool with_side_inputs)
{
  srand(time(0));
  const int num_layers = random(1, 5);

  Hash** hashes = new Hash*[num_layers];
  if (!hashes) { return eIcicleError::ALLOCATION_FAILED; }

  const int min_layer_stored = random(0, num_layers);
  std::cout << "Tree has " << num_layers << " Layers with the following hash layout (Min stored layer: " << min_layer_stored << "):\n";
  int num_outputs = 1;
  int num_inputs = 1;
  int num_side_inputs = 0;
  for (int i = 0; i < num_layers; i++)
  {
    num_inputs =  random(1,5) * num_outputs; // Ensuring outputs of layer i divide inputs of layer i+1
    num_outputs = random(1, num_inputs + 1);
    if (with_side_inputs) { num_side_inputs = random(0, 5); }
    hashes[i] = new SimpleHash(num_inputs, num_outputs, num_side_inputs);
    if (!hashes[i]) { return eIcicleError::ALLOCATION_FAILED; }

    std::cout << "IN=" << num_inputs << "\tOUT=" << num_outputs << "\tSIDE=" << num_side_inputs << '\n';
  }

  int num_hashes_in_layer = 1;
  int total_side_limbs_in_tree = 0;
  for (int i = num_layers - 1; i > 0; i--)
  {
    total_side_limbs_in_tree += num_hashes_in_layer * hashes[i]->m_total_secondary_input_limbs;
    num_hashes_in_layer *= hashes[i]->m_total_input_limbs / hashes[i - 1]->m_total_output_limbs;
  }
  total_side_limbs_in_tree += num_hashes_in_layer * hashes[0]->m_total_secondary_input_limbs;

  int total_input_limbs = num_hashes_in_layer * hashes[0]->m_total_input_limbs;

  int leaf_size = total_input_limbs + 1;
  while (total_input_limbs % leaf_size != 0 || hashes[0]->m_total_input_limbs % leaf_size != 0) {
    leaf_size = random(1, total_input_limbs + 1);
  }

  MerkleTree tree = MerkleTree(num_layers, (const Hash**) hashes, leaf_size, min_layer_stored);
  std::cout << "leaf size:\t" << leaf_size << '\n';
  std::cout << "Building a tree from the following " << (total_input_limbs / leaf_size) << " leaves:\n";  
  limb_t* inputs = new limb_t[total_input_limbs];
  for (size_t i = 0; i < total_input_limbs; i++) { 
    inputs[i] = random(0, (1 << 10) - 1); std::cout << inputs[i] << '\t'; 
  }

  limb_t* side_inputs = nullptr;
  if (with_side_inputs) {
    std::cout << "\n\nAnd the following " << total_side_limbs_in_tree << " side inputs:\n";
    side_inputs = new limb_t[total_side_limbs_in_tree];
    for (size_t i = 0; i < total_side_limbs_in_tree; i++) { 
      side_inputs[i] = random(0, (1 << 10) - 1); std::cout << side_inputs[i] << '\t'; 
    }
  }

  std::cout << "\n\n\nTree:\n";
  tree.build(inputs, MerkleTreeConfig(), side_inputs);
  tree.print_tree();
  std::cout << "Validating tree build - ";
  assert_valid_tree(&tree, num_layers, total_input_limbs, inputs, side_inputs, (const Hash**) hashes, min_layer_stored);
  std::cout << "V\n\n";

  if (!with_side_inputs) {
    const int path_element_idx = random(0, total_input_limbs / leaf_size);

    std::cout << "V\n\nGetting path of the following index: " << path_element_idx << '\n';

    // Check path
    unsigned num_limbs_in_path = hashes[num_layers-1]->m_total_output_limbs;
    for (size_t i = 0; i < num_layers; i++) { num_limbs_in_path += hashes[i]->m_total_input_limbs; }

    limb_t* path;
    if (tree.allocate_path(path, num_limbs_in_path) != eIcicleError::SUCCESS) { return eIcicleError::ALLOCATION_FAILED; }
    std::cout << "Path size: " << num_limbs_in_path << '\n';
    ICICLE_ASSERT(tree.get_path(inputs, path_element_idx, path, MerkleTreeConfig()) == eIcicleError::SUCCESS) 
      << "Couldn't get path\n";
    tree.print_path(path);

    std::cout << "Validating path (both library validation and naive validation) - ";
    bool verification_valid = false;
    ICICLE_ASSERT(tree.verify(path, path_element_idx, verification_valid, MerkleTreeConfig()) == eIcicleError::SUCCESS)
      << "Couldn't verify\n";
    ICICLE_ASSERT(verification_valid) << "Verification failed.\n";

    assert_valid_path(path, path_element_idx * leaf_size, num_layers, (const Hash**) hashes);
    std::cout << "V\n";
  }

  delete[] inputs;
  delete[] side_inputs;
  for (int i = 0; i < num_layers; i++) { delete hashes[i]; }
  delete[] hashes;

  return eIcicleError::SUCCESS;
}

int main()
{
  while(1) assert(tree_test(1) == eIcicleError::SUCCESS);
}