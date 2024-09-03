#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "hash.h"
#include "hashes.h"
#include "icicle/utils/utils.h"
#include "merkle_tree.h"

#include <cassert>

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
  // COMMENT Currently does not support hashes with side inputs
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

int main()
{
  const int nof_layers = 4;
  const int leaf_size = 1;
  const int output_store_min_layer = 0;
  const int path_element_idx = 1;
  MerkleTreeConfig config = MerkleTreeConfig();

  Add6to2OutputsHash* tri_hash =      new Add6to2OutputsHash();
  AddHash* two_hash =                 new AddHash();
  Add2HashWithSideInput* side_hash =  new Add2HashWithSideInput();
  // const Hash* hashes[nof_layers] = {side_hash, tri_hash, tri_hash, two_hash};
  const Hash* hashes[nof_layers] = {two_hash, two_hash, two_hash, tri_hash};

  int num_inputs = 1;
  for (int i = nof_layers - 1; i >= 0; i--)
  {
    const u_int64_t lower_layer_out_limbs = (i > 0) ? hashes[i-1]->m_total_output_limbs : 1;
    num_inputs *= hashes[i]->m_total_input_limbs / lower_layer_out_limbs;
  }
  
  std::cout << "Building a tree from the following " << num_inputs << " leaves:\n";
  limb_t* inputs = new limb_t[num_inputs];
  for (size_t i = 0; i < num_inputs; i++) { inputs[i] = i; std::cout << inputs[i] << '\t'; }

  int num_side_inputs = 0;
  int num_hashes_in_layer = 1;
  for (int i = nof_layers - 1; i >= 0; i--)
  {
    num_side_inputs += hashes[i]->m_total_secondary_input_limbs * num_hashes_in_layer;
    const u_int64_t lower_layer_out_limbs = (i > 0) ? hashes[i-1]->m_total_output_limbs : 1;
    num_hashes_in_layer *= hashes[i]->m_total_input_limbs / lower_layer_out_limbs;
  }

  limb_t* side_ins = num_side_inputs > 0? new limb_t[num_side_inputs] : nullptr;
  if (num_side_inputs > 0) { std::cout << "\nSide inputs:\n"; }
  for (size_t i = 0; i < num_side_inputs; i++) { side_ins[i] = i; std::cout << side_ins[i] << '\t'; }

  MerkleTree* tree = new MerkleTree(nof_layers, hashes, leaf_size, output_store_min_layer);
  std::cout << "\n\nTree:\n";
  tree->build(inputs, config, side_ins);
  tree->print_tree();

  // Check valid tree calc
  std::cout << "Checking the tree was correctly built - ";
  assert_valid_tree(tree, nof_layers, num_inputs, inputs, side_ins, hashes, output_store_min_layer);
  std::cout << "V\n\nGetting path of the following index: " << path_element_idx << '\n';

  // Check path
  unsigned nof_limbs_in_path = num_inputs;
  for (size_t i = 0; i < nof_layers; i++) { nof_limbs_in_path += hashes[i]->m_total_output_limbs; }

  limb_t* path;
  assert(tree->allocate_path(path, nof_limbs_in_path) == eIcicleError::SUCCESS);
  assert(tree->get_path(inputs, path_element_idx, path, config) == eIcicleError::SUCCESS);
  tree->print_path(path);

  std::cout << "Validating path (both library validation and naive validation) - ";
  bool verification_valid = false;
  assert(tree->verify(path, path_element_idx, verification_valid, config) == eIcicleError::SUCCESS);
  assert(verification_valid);

  assert_valid_path(path, path_element_idx * leaf_size, nof_layers, hashes);
  std::cout << "V\n";

  delete[] inputs;
  delete tri_hash;
  delete two_hash;
  delete tree;
  if (num_side_inputs > 0) { delete[] side_ins; }
}