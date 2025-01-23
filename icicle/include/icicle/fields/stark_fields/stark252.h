#pragma once

#include "icicle/fields/storage.h"
#include "icicle/fields/field.h"
#include "icicle/fields/params_gen.h"

// modulus = 3618502788666131213697322783095070105623107215331596699973092056135872020481 (2^251+17*2^192+1)

namespace stark252 {
  struct fp_config {
    static constexpr storage<8> modulus = {0x00000001, 0x00000000, 0x00000000, 0x00000000,
                                           0x00000000, 0x00000000, 0x00000011, 0x08000000};

    static constexpr storage_array<3, 8> reduced_digits = {
      {{0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
       {0xffffffe1, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xfffffdf0, 0x07ffffff},
       {0x7e000401, 0xfffffd73, 0x330fffff, 0x00000001, 0xff6f8000, 0xffffffff, 0x5e008810, 0x07ffd4ab}}};
    static constexpr unsigned reduced_digits_count = 3;
    PARAMS(modulus)
    MOD_SQR_SUBS()

    static constexpr storage<8> rou = {0x42f8ef94, 0x6070024f, 0xe11a6161, 0xad187148,
                                       0x9c8b0fa5, 0x3f046451, 0x87529cfa, 0x005282db};
    TWIDDLES(modulus, rou)
  };

  /**
   * Scalar field. Is always a prime field.
   */
  typedef Field<fp_config> scalar_t;
} // namespace stark252