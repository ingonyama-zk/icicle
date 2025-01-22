#pragma once

#include "icicle/fields/storage.h"
#include "icicle/fields/field.h"
#include "icicle/fields/params_gen.h"

namespace bls12_377 {
  struct fp_config {
    static constexpr storage<8> modulus = {0x00000001, 0x0a118000, 0xd0000001, 0x59aa76fe,
                                           0x5c37b001, 0x60b44d1e, 0x9a2ca556, 0x12ab655e};
    static constexpr storage_array<3, 8> reduced_digits = {
      {{0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
       {0xfffffff3, 0x7d1c7fff, 0x6ffffff2, 0x7257f50f, 0x512c0fee, 0x16d81575, 0x2bbb9a9d, 0x0d4bda32},
       {0xb861857b, 0x25d577ba, 0x8860591f, 0xcc2c27b5, 0xe5dc8593, 0xa7cc008f, 0xeff1c939, 0x011fdae7}}};
    static constexpr unsigned reduced_digits_count = 3;
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
