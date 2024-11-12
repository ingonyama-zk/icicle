#pragma once

#include "icicle/fields/storage.h"
#include "icicle/fields/field.h"
#include "icicle/fields/params_gen.h"

namespace bls12_377 {
  struct fp_config {
    static constexpr storage<8> modulus = {0x00000001, 0x0a118000, 0xd0000001, 0x59aa76fe,
                                           0x5c37b001, 0x60b44d1e, 0x9a2ca556, 0x12ab655e};
    // static constexpr storage<8> mont_inv_modulus = {0xffffffff, 0xa117fff, 0x90000001, 0x452217cc, 0x4790a000, 0x249765c3, 0x68b29556, 0x6992d0fa};
    PARAMS(modulus)

    static constexpr storage<8> rou = {0xec2a895e, 0x476ef4a4, 0x63e3f04a, 0x9b506ee3,
                                       0xd1a8a12f, 0x60c69477, 0x0cb92cc1, 0x11d4b7f6};
    TWIDDLES(modulus, rou)
  };

  /**
   * Scalar field. Is always a prime field.
   */
  typedef Field<fp_config> scalar_t;
} // namespace bls12_377
