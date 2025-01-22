#pragma once

#include "icicle/fields/storage.h"
#include "icicle/fields/params_gen.h"

namespace bls12_377 {
  struct fq_config {
    static constexpr storage<12> modulus = {0x00000001, 0x8508c000, 0x30000000, 0x170b5d44, 0xba094800, 0x1ef3622f,
                                            0x00f5138f, 0x1a22d9f3, 0x6ca1493b, 0xc63b05c0, 0x17c510ea, 0x01ae3a46};
    static constexpr storage_array<2, 12> reduced_digits = {
      {{0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000},
       {0xffffff68, 0x02cdffff, 0x7fffffb1, 0x51409f83, 0x8a7d3ff2, 0x9f7db3a9, 0x6e7c6305, 0x7b4e97b7, 0x803c84e8,
        0x4cf495bf, 0xe2fdf49a, 0x008d6661}}};
    static constexpr unsigned reduced_digits_count = 2;
    PARAMS(modulus)

    static constexpr storage<12> rou = {0xc563b9a1, 0x7eca603c, 0x06fe0bc3, 0x06df0a43, 0x0ddff8c6, 0xb44d994a,
                                        0x4512a3d4, 0x40fbe05b, 0x8aeffc9b, 0x30f15248, 0x05198a80, 0x0036a92e};
    TWIDDLES(modulus, rou)

    // nonresidue to generate the extension field
    static constexpr uint32_t nonresidue = 5;
    // true if nonresidue is negative
    static constexpr bool nonresidue_is_negative = true;
  };
} // namespace bls12_377
