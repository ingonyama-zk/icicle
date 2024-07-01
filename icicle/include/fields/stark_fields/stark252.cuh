#pragma once

#include "fields/storage.cuh"
#include "fields/field.cuh"
#include "fields/params_gen.cuh"

// modulus = 3618502788666131213697322783095070105623107215331596699973092056135872020481 (2^251+17*2^192+1)
namespace stark252 {
  struct fp_config {
    static constexpr storage<8> modulus = {0x00000001, 0x00000000, 0x00000000, 0x00000000,
                                           0x00000000, 0x00000000, 0x00000011, 0x08000000};
    PARAMS(modulus)

    static constexpr storage<8> rou = {0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                       0x00000000, 0x00000000, 0x00000011, 0x08000000};
    TWIDDLES(modulus, rou)
  };

  /**
   * Scalar field. Is always a prime field.
   */
  typedef Field<fp_config> scalar_t;
} // namespace stark252