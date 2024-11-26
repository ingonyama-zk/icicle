#pragma once

#include <functional>
#include "icicle/utils/utils.h"
#include "icicle/device.h"
#include "icicle/hash/poseidon2.h"

#include "icicle/fields/field_config.h"
using namespace field_config;

namespace icicle {

  // Options for generating the on-device Poseidon2 constants.
  // This struct will hold parameters needed to initialize Poseidon2 constants with custom settings.
  // The fields will include:
  // - `t`: The t (branching factor) of the Poseidon2 hash.
  // - `alpha`: The exponent used in the S-box function.
  // - `nof_rounds`: The number of rounds (both full and partial) for the Poseidon2 hash.
  // - `mds_matrix`: The Maximum Distance Separable (MDS) matrix used for mixing the state.
  // The struct should be FFI (Foreign Function Interface) compatible, meaning it should use basic types or pointers
  // that can easily be shared between languages. The template parameter `S` represents the field type for which the
  // Poseidon2 constants are being initialized.
  template <typename S>
  struct Poseidon2ConstantsOptions {
    unsigned int t = 0;
    unsigned int alpha = 5;              ///< Sbox power.
    bool use_all_zeroes_padding = true;  ///< If true use [0,0,..,0] for padding. Otherwise use [1,0,..,0].
    unsigned int nof_upper_full_rounds;  ///< Number of upper full rounds of a single hash.
    unsigned int nof_partial_rounds;     ///< Number of partial rounds of a single hash.
    unsigned int nof_bottom_full_rounds; ///< Number of bottom full rounds of a single hash.
    S* rounds_constants; ///< Round constants (both of the full and the partial rounds). The order of the constants in
                         ///< the memory is according to the rounds order.
    S* mds_matrix;       ///> MDS matrix is used in the full rounds. The same matrix is used for all such rounds.
    // S* partial_matrix_diagonal;   ///< Partial matrix is used in the partial rounds. The same matrix is used for all such rounds.
    //                               ///< Only M[i,i] member are different from 1. These members are here.
    S* partial_matrix_diagonal_m1;   ///< This partial matrix is used in the partial rounds instead of partial_matrix_diagonal 
                                  ///< (using this matrix improves the performance of the partial rounds). The same matrix
                                  ///< is used for all such rounds. Only M[i,i] member are different from 1. 
                                  ///< These members are here. 
    Poseidon2ConstantsOptions() {}
    Poseidon2ConstantsOptions(const Poseidon2ConstantsOptions&) = delete;
  };

  /*************************** Backend registration ***************************/

  // Note: 'phantom' is a workaround for the function required per field but need to differentiate by type when
  // calling.

  using CreatePoseidon2Impl = std::function<eIcicleError(
    const Device& device, unsigned t, const scalar_t* domain_tag, std::shared_ptr<HashBackend>& /*OUT*/)>;

  // poseidon2 init constants
  void register_create_poseidon2(const std::string& deviceType, CreatePoseidon2Impl impl);

#define REGISTER_CREATE_POSEIDON2_BACKEND(DEVICE_TYPE, FUNC)                                                            \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_create_poseidon2) = []() -> bool {                                                          \
      register_create_poseidon2(DEVICE_TYPE, FUNC);                                                                     \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

} // namespace icicle