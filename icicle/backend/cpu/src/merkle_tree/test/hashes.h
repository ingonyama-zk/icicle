#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "hash.h"
#include "icicle/utils/utils.h"

class SimpleHash : public Hash
{
public:
  SimpleHash(const int nof_inputs, const int nof_outputs, const int nof_sides) : 
    Hash(nof_inputs, nof_outputs, nof_sides) {}
  
  eIcicleError
  run_single_hash(
    const limb_t* input_limbs,
    limb_t* output_limbs, 
    const HashConfig& config,
    const limb_t* secondary_input_limbs = nullptr) const
  {
    for (int i = 0; i < m_total_output_limbs; i++)
    {
      output_limbs[i] = input_limbs[0];
      for (int j = 1; j < m_total_input_limbs; j++)
      {
        output_limbs[i] += input_limbs[j];
      }
      for (int j = 0; j < m_total_secondary_input_limbs; j++)
      {
        output_limbs[i] += secondary_input_limbs[j];
      }
      
      output_limbs[i] *= (i+1);
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
      run_single_hash(&input_limbs[i*m_total_input_limbs], &output_limbs[i*m_total_output_limbs], config, side_input_limbs);
    }
    return eIcicleError::SUCCESS;
  }
};
