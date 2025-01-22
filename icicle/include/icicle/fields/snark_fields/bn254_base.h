#pragma once

#include "icicle/fields/storage.h"
#include "icicle/fields/params_gen.h"

namespace bn254 {
  struct fq_config {
    static constexpr storage<8> modulus = {0xd87cfd47, 0x3c208c16, 0x6871ca8d, 0x97816a91,
                                           0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
    static constexpr storage_array<3, 8> reduced_digits = {
      {{0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
       {0xc58f0d9d, 0xd35d438d, 0xf5c70b3d, 0x0a78eb28, 0x7879462c, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1},
       {0x538afa89, 0xf32cfc5b, 0xd44501fb, 0xb5e71911, 0x0a417ff6, 0x47ab1eff, 0xcab8351f, 0x06d89f71}}};
    static constexpr unsigned reduced_digits_count = 3;

    PARAMS(modulus)

    // nonresidue to generate the extension field
    static constexpr uint32_t nonresidue = 1;
    // true if nonresidue is negative
    static constexpr bool nonresidue_is_negative = true;
  };
} // namespace bn254
