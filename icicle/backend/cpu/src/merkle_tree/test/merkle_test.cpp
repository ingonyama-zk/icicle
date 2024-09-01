#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "hash.h"
#include "icicle/utils/utils.h"
#include "merkle_tree.h"

#include <cassert>

// TODO test with side limbs
class SimpleHash : public Hash
{
public:
  SimpleHash(): Hash(2, 1, 0) {}

  eIcicleError
  run_single_hash(const limb_t* input_limbs, limb_t* output_limbs, const HashConfig& config, const limb_t* secondary_input_limbs = nullptr) const
  {
    for (int i = 0; i < m_total_output_limbs; i++)
    {
      output_limbs[i] = input_limbs[0];
      for (int j = 1; j < m_total_input_limbs; j++)
      {
        output_limbs[i] += input_limbs[j];
      }
    }
    return eIcicleError::SUCCESS;
  }

  eIcicleError 
  run_multiple_hash(
      const limb_t* input_limbs,
      limb_t* output_limbs,
      int nof_hashes,
      const HashConfig& config,
      const limb_t* side_input_limbs = nullptr) const
  {
    for (int i = 0; i < nof_hashes; i++)
    {
      run_single_hash(&input_limbs[i*m_total_input_limbs], &output_limbs[i*m_total_output_limbs], config);
    }
    return eIcicleError::SUCCESS;
  }
};

void assert_valid_tree(
  const MerkleTree* tree,
  const int nof_layers,
  int nof_inputs,
  const limb_t* inputs,
  const limb_t* side_inputs,
  const Hash* hashes,
  const int output_store_min_layer
)
{ 
  int nof_outputs = nof_inputs * hashes[0].m_total_output_limbs / hashes[0].m_total_input_limbs;
  limb_t* layer_in =  new limb_t[nof_inputs];   // Going layer by layer and having the input layer as the largest 
  limb_t* layer_out = new limb_t[nof_outputs];  // ensures  these are the maximum sizes required for inputs and outputs
  // NOTE there is an assumption here that output number is less or equal to input number for all layers

  for (int i = 0; i < nof_inputs; i++) { layer_in[i] = inputs[i]; }
  
  int nof_hashes = nof_inputs / hashes[0].m_total_input_limbs;
  int side_inputs_offset = 0;
  for (int i = 0; i < nof_layers; i++)
  {
    nof_outputs = nof_inputs * hashes[i].m_total_output_limbs / hashes[i].m_total_input_limbs;
    // TODO actual config to allow gpu test as well?
    hashes[i].run_multiple_hash(layer_in, layer_out, nof_hashes, HashConfig(), &side_inputs[side_inputs_offset]);

    // Assert stored values in the trees (if they exist)
    int nof_hashes = nof_inputs / hashes[i].m_total_input_limbs;
    
    if (i >= output_store_min_layer) {
      for (int j = 0; j < nof_hashes; j++) { 
        const limb_t* result = nullptr;
        tree->get_hash_result(i, j, result);
        assert(layer_out[j] == *result);
      }
    }

    // Transfer outputs to inputs before moving to the next layer
    for (int i = 0; i < nof_outputs; i++) { layer_in[i] = layer_out[i]; }
    nof_inputs = nof_outputs;
    side_inputs_offset += hashes[i].m_total_secondary_input_limbs;
  }

  // Assert final root
  const limb_t* root;
  tree->get_root(root);
  for (int i = 0; i < hashes[nof_layers-1].m_total_output_limbs; i++) { assert(root[i] == layer_out[i]); }

  delete[] layer_in;
  delete[] layer_out;
}

void assert_valid_path(
  const limb_t* path, const int nof_layers, const Hash* hashes
)
{
  // COMMENT Currently does not support hashes with side inputs
  int input_offset = 0;
  int output_offset = 0;
  for (int i = 0; i < nof_layers; i++)
  {
    limb_t* hash_results = new limb_t[hashes[i].m_total_output_limbs];
    output_offset += hashes[i].m_total_input_limbs;
    hashes[i].run_single_hash(&path[input_offset], hash_results, HashConfig());
    // Compare calculated results with the path
    for (int j = 0; j < hashes[i].m_total_output_limbs; j++) { assert(hash_results[j] == path[output_offset + j]); }
    
    input_offset = output_offset;
    delete[] hash_results;
  }
}

int main()
{
  int nof_layers = 4;
  int leaf_size = 1;
  int output_store_min_layer = 0; // Only top currently
  MerkleTreeConfig config;

  SimpleHash* hashes = new SimpleHash[nof_layers];
  const int num_inputs = 16;
  limb_t* inputs = new limb_t[num_inputs];
  for (size_t i = 0; i < num_inputs; i++) { inputs[i] = i; std::cout << inputs[i] << '\t'; }
  std::cout << '\n';

  int num_side_inputs = 0;
  int num_hashes_in_layer = num_inputs;
  for (int i = 0; i < nof_layers; i++)
  {
    num_hashes_in_layer /= hashes[i].m_total_input_limbs;
    num_side_inputs = hashes[i].m_total_secondary_input_limbs * num_hashes_in_layer;
  }

  limb_t* side_ins = num_side_inputs > 0? new limb_t[num_side_inputs] : nullptr;
  for (size_t i = 0; i < num_side_inputs; i++) { side_ins[i] = i; }

  MerkleTree* tree = new MerkleTree(nof_layers, hashes, leaf_size, output_store_min_layer);
  tree->build(inputs, MerkleTreeConfig(), side_ins);
  tree->print_tree();

  // Check valid tree calc
  assert_valid_tree(tree, nof_layers, num_inputs, inputs, side_ins, hashes, output_store_min_layer);

  // Check path
  unsigned nof_limbs_in_path = num_inputs;
  for (size_t i = 0; i < nof_layers; i++) { nof_limbs_in_path += hashes[i].m_total_output_limbs; }

  limb_t* path;
  assert(tree->allocate_path(path, nof_limbs_in_path) == eIcicleError::SUCCESS);
  assert(tree->get_path(inputs, 0, path, MerkleTreeConfig()) == eIcicleError::SUCCESS);

  bool verification_valid = false;
  assert(tree->verify(path, 0, verification_valid, MerkleTreeConfig()) == eIcicleError::SUCCESS);
  assert(verification_valid);
  tree->print_path(path);

  assert_valid_path(path, nof_layers, hashes);


  delete[] hashes;
  delete[] inputs;
  delete tree;
  if (num_side_inputs > 0) { delete[] side_ins; }
}