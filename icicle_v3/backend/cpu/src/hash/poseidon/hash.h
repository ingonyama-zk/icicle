#pragma once

#include "icicle/errors.h"
#include "icicle/runtime.h"
#include "icicle/utils/utils.h"
// #include "icicle/config_extension.h"

#include <cstdint>
#include <functional>

typedef uint32_t limb_t;

/*************************** Frontend & Backend shared APIs ***************************/
namespace icicle {


  /**
   * @brief Configuration struct for the hash.
   */
  struct HashConfig {
    bool are_inputs_on_device = false;  ///< True if inputs are on device, false if on host. Default is false.
    bool are_outputs_on_device = false; ///< True if outputs are on device, false if on host. Default is false.
    bool is_async = false; ///< True to run the hash asynchronously, false to run it synchronously. Default is false.
    // ConfigExtension* ext = nullptr; ///< Backend-specific extensions.
  };

  /**
   * @brief Abstract class representing a hash function. In order to support structures like mmcs, we support
   * side-inputs to the hash. These are inputs that are inserted in an intermediate stage of the calculations.
   */
  class Hash
  {
  public:
    const int total_input_limbs;      // Total number of regular input limbs
    const int total_output_limbs;     // Total number of output limbs
    const int total_side_input_limbs; // Total number of side input limbs

    // Constructor
    Hash(int total_input_limbs, int total_output_limbs, int total_side_input_limbs = 0)
        : total_input_limbs(total_input_limbs), total_output_limbs(total_output_limbs),
          total_side_input_limbs(total_side_input_limbs)
    {
    }

    /**
     * @brief Pure virtual function to run a single hash.
     * @param input_limbs Pointer to the input limbs.
     * @param output_limbs Pointer to the output limbs.
     * @return Error code of type eIcicleError.
     */
    virtual eIcicleError
    run_single_hash(const limb_t* input_limbs, limb_t* output_limbs, const HashConfig& config) const = 0;

    /**
     * @brief Pure virtual function to run multiple hashes.
     * @param input_limbs Pointer to the input limbs.
     * @param output_limbs Pointer to the output limbs.
     * @param nof_hashes Number of hashes to run.
     * @return Error code of type eIcicleError.
     */
    virtual eIcicleError run_multiple_hash(
      const limb_t* input_limbs,
      limb_t* output_limbs,
      int nof_hashes,
      const HashConfig& config,
      const limb_t* side_input_limbs = nullptr) const = 0;
  };

}; // namespace icicle

