#pragma once

#include "icicle/fields/storage.h"
#include "icicle/fields/field.h"
#include "icicle/fields/params_gen.h"

namespace bn254 {
  struct fp_config {
    static constexpr storage<8> modulus = {0xf0000001, 0x43e1f593, 0x79b97091, 0x2833e848,
                                           0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
static constexpr storage_array<3, 8> reduced_digits = {{
{0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
{0x4ffffffb, 0xac96341c, 0x9f60cd29, 0x36fc7695, 0x7879462e, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1},
{0xae216da7, 0x1bb8e645, 0xe35c59e3, 0x53fe3ab1, 0x53bb8085, 0x8c49833d, 0x7f4e44a5, 0x0216d0b1}}};
    static constexpr unsigned reduced_digits_count = 3;
    PARAMS(modulus)
    MOD_SQR_SUBS()

    static constexpr storage<8> rou = {0x725b19f0, 0x9bd61b6e, 0x41112ed4, 0x402d111e,
                                       0x8ef62abc, 0x00e0a7eb, 0xa58a7e85, 0x2a3c09f0};
    TWIDDLES(modulus, rou)
  };

  /**
   * Scalar field. Is always a prime field.
   */
  typedef Field<fp_config> scalar_t;
} // namespace bn254
