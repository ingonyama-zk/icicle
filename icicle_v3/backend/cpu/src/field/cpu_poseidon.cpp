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
    int input_pointer = 0;
    int output_pointer = 0;
    int output_size = this->element_nof_limbs * this->output_nof_elements;
    int input_size = this->element_nof_limbs * this->input_nof_elements;
    for (int batch_idx = 0; batch_idx<batch; batch_idx++)
    {
      std::copy(input_limbs + batch_idx*input_size, input_limbs + batch_idx*input_size + output_size, output_limbs + batch_idx*output_size);
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