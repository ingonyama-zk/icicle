#include <iostream>
#include "icicle/hash.h"
#include "icicle/errors.h"

// #include "../../include/icicle/hash.h"
using namespace icicle;

class Poseidon : public Hash {
 public:
  Poseidon(int element_nof_limbs, int input_nof_elements, int output_nof_elements)
      : Hash(element_nof_limbs, input_nof_elements, output_nof_elements) {}

  virtual eIcicleError hash_many(const limb_t *input_limbs, limb_t *output_limbs, unsigned int batch) const override {
    // This is just a placeholder, copy first element of each hash group.
    for (int batch_idx = 0; batch_idx<batch; batch_idx++)
    {
      for (int output_element_idx = 0; output_element_idx < this->output_nof_elements; output_element_idx++)
      {
          for (int limb_idx = 0; limb_idx < element_nof_limbs; limb_idx++)
          {
             int output_pointer = batch_idx* this->output_nof_elements*this->element_nof_limbs + output_element_idx*this->element_nof_limbs + limb_idx;
             int input_pointer = batch_idx * this->input_nof_elements * this->element_nof_limbs;
             output_limbs[output_pointer] = input_limbs[input_pointer];
          }
      }
    }
    return eIcicleError::SUCCESS;
  }
};





eIcicleError poseidon_cpu(const Device& device, Hash** hash, int element_nof_limbs, int input_nof_elements, int output_nof_elements)
{
    *hash = new Poseidon(element_nof_limbs, input_nof_elements, output_nof_elements);
    return eIcicleError::SUCCESS;
}

REGISTER_POSEIDON_BACKEND("CPU", poseidon_cpu);