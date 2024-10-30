#pragma once

#include <functional>
#include "icicle/utils/utils.h"
#include "icicle/device.h"
#include "icicle/hash/poseidon.h"

#include "icicle/fields/field_config.h"
using namespace field_config;

namespace icicle {

  // Options for generating the on-device Poseidon constants.
  // This struct will hold parameters needed to initialize Poseidon constants with custom settings.
  // The fields will include:
  // - `t`: The t (branching factor) of the Poseidon hash.
  // - `alpha`: The exponent used in the S-box function.
  // - `nof_rounds`: The number of rounds (both full and partial) for the Poseidon hash.
  // - `mds_matrix`: The Maximum Distance Separable (MDS) matrix used for mixing the state.
  // The struct should be FFI (Foreign Function Interface) compatible, meaning it should use basic types or pointers
  // that can easily be shared between languages. The template parameter `S` represents the field type for which the
  // Poseidon constants are being initialized.
  template <typename S>
  struct PoseidonConstantsOptions {
    unsigned int t = 0;
    unsigned int alpha;                 ///< Sbox power.
    bool use_domain_tag = false;        ///< If i_domain_tag is set then single hash width = t + 1, otherwise width = t.
    S domain_tag_value = S::zero();     ///< Domain tag value that is usually used in sponge function Poseidon hashes.
    bool use_all_zeroes_padding = true; ///< If true use [0,0,..,0] for padding. Otherwise use [1,0,..,0].
    unsigned int nof_upper_full_rounds; ///< Number of upper full rounds of a single hash.
    unsigned int nof_partial_rounds;    ///< Number of partial rounds of a single hash.
    unsigned int nof_bottom_full_rounds; ///< Number of bottom full rounds of a single hash.
    S* rounds_constants; ///< Round constants (both of the full and the partial rounds). The order of the constants in
                         ///< the memory is according to the rounds order.
    S* mds_matrix;       ///> MDS matrix used in the full rounds. The same matrix is used for all the full rounds.
    S* pre_matrix;       ///< Pre-matrix used in the last upper full round.
    S* sparse_matrices; ///< Sparse matries that are used in the partial rounds. A single aprse matrix in the memory has
                        ///< "t x t" members. The calculation is done only on the member that not equal to zero.
  };

  /*************************** Backend registration ***************************/

  // Note: 'phantom' is a workaround for the function required per field but need to differentiate by type when
  // calling.

  using CreatePoseidonImpl = std::function<eIcicleError(
    const Device& device,
    unsigned t,
    bool use_domain_tag,
    std::shared_ptr<HashBackend>& /*OUT*/,
    const scalar_t& phantom)>;

  // poseidon init constants
  void register_create_poseidon(const std::string& deviceType, CreatePoseidonImpl impl);

#define REGISTER_CREATE_POSEIDON_BACKEND(DEVICE_TYPE, FUNC)                                                            \
  namespace {                                                                                                          \
    static bool UNIQUE(_reg_create_poseidon) = []() -> bool {                                                          \
      register_create_poseidon(DEVICE_TYPE, FUNC);                                                                     \
      return true;                                                                                                     \
    }();                                                                                                               \
  }

} // namespace icicle