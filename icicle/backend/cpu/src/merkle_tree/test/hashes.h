#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "hash.h"
#include "icicle/utils/utils.h"

class Add6to2OutputsHash : public Hash
{
public:
  Add6to2OutputsHash(): Hash(6, 2, 0) {}

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

class AddHash : public Hash
{
public:
  AddHash(): Hash(2, 1, 0) {}

  eIcicleError
  run_single_hash(const limb_t* input_limbs, limb_t* output_limbs, const HashConfig& config, const limb_t* secondary_input_limbs = nullptr) const
  {
    output_limbs[0] = input_limbs[0];
    for (int j = 1; j < m_total_input_limbs; j++)
    {
      output_limbs[0] += input_limbs[j];
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

class Add2HashWithSideInput : public Hash
{
public:
  Add2HashWithSideInput(): Hash(2, 1, 1) {}

  eIcicleError
  run_single_hash(const limb_t* input_limbs, limb_t* output_limbs, const HashConfig& config, const limb_t* secondary_input_limbs = nullptr) const
  {
    output_limbs[0] = input_limbs[0];
    for (int j = 1; j < m_total_input_limbs; j++)
    {
      output_limbs[0] += input_limbs[j];
    }
    for (int j = 0; j < m_total_secondary_input_limbs; j++)
    {
      output_limbs[0] += secondary_input_limbs[j];
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
    std::cout << "Nof hashes = " << nof_hashes << '\n';
    for (int i = 0; i < nof_hashes; i++)
    {
      std::cout << "Offsets:\t" << i*m_total_input_limbs << '\t' << i*m_total_output_limbs << '\t' << i*m_total_secondary_input_limbs << '\n';
      run_single_hash(&input_limbs[i*m_total_input_limbs], &output_limbs[i*m_total_output_limbs], config, 
        &side_input_limbs[i*m_total_secondary_input_limbs]);
    }
    return eIcicleError::SUCCESS;
  }
};
