#pragma once
#ifndef BLS12_381_BASE_PARAMS_H
#define BLS12_381_BASE_PARAMS_H

#include "fields/storage.h"

namespace bls12_381 {
  struct fq_config {
    static constexpr unsigned limbs_count = 12;
    static constexpr unsigned modulus_bit_count = 381;
    static constexpr unsigned num_of_reductions = 1;
    static constexpr storage<limbs_count> modulus = {0xffffaaab, 0xb9feffff, 0xb153ffff, 0x1eabfffe,
                                                     0xf6b0f624, 0x6730d2a0, 0xf38512bf, 0x64774b84,
                                                     0x434bacd7, 0x4b1ba7b6, 0x397fe69a, 0x1a0111ea};
    static constexpr storage<limbs_count> modulus_2 = {0xffff5556, 0x73fdffff, 0x62a7ffff, 0x3d57fffd,
                                                       0xed61ec48, 0xce61a541, 0xe70a257e, 0xc8ee9709,
                                                       0x869759ae, 0x96374f6c, 0x72ffcd34, 0x340223d4};
    static constexpr storage<limbs_count> modulus_4 = {0xfffeaaac, 0xe7fbffff, 0xc54ffffe, 0x7aaffffa,
                                                       0xdac3d890, 0x9cc34a83, 0xce144afd, 0x91dd2e13,
                                                       0x0d2eb35d, 0x2c6e9ed9, 0xe5ff9a69, 0x680447a8};
    static constexpr storage<limbs_count> neg_modulus = {0x00005555, 0x46010000, 0x4eac0000, 0xe1540001,
                                                         0x094f09db, 0x98cf2d5f, 0x0c7aed40, 0x9b88b47b,
                                                         0xbcb45328, 0xb4e45849, 0xc6801965, 0xe5feee15};
    static constexpr storage<2 * limbs_count> modulus_wide = {
      0xffffaaab, 0xb9feffff, 0xb153ffff, 0x1eabfffe, 0xf6b0f624, 0x6730d2a0, 0xf38512bf, 0x64774b84,
      0x434bacd7, 0x4b1ba7b6, 0x397fe69a, 0x1a0111ea, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<2 * limbs_count> modulus_squared = {
      0x1c718e39, 0x26aa0000, 0x76382eab, 0x7ced6b1d, 0x62113cfd, 0x162c3383, 0x3e71b743, 0x66bf91ed,
      0x7091a049, 0x292e85a8, 0x86185c7b, 0x1d68619c, 0x0978ef01, 0xf5314933, 0x16ddca6e, 0x50a62cfd,
      0x349e8bd0, 0x66e59e49, 0x0e7046b4, 0xe2dc90e5, 0xa22f25e9, 0x4bd278ea, 0xb8c35fc7, 0x02a437a4};
    static constexpr storage<2 * limbs_count> modulus_squared_2 = {
      0x38e31c72, 0x4d540000, 0xec705d56, 0xf9dad63a, 0xc42279fa, 0x2c586706, 0x7ce36e86, 0xcd7f23da,
      0xe1234092, 0x525d0b50, 0x0c30b8f6, 0x3ad0c339, 0x12f1de02, 0xea629266, 0x2dbb94dd, 0xa14c59fa,
      0x693d17a0, 0xcdcb3c92, 0x1ce08d68, 0xc5b921ca, 0x445e4bd3, 0x97a4f1d5, 0x7186bf8e, 0x05486f49};
    static constexpr storage<2 * limbs_count> modulus_squared_4 = {
      0x71c638e4, 0x9aa80000, 0xd8e0baac, 0xf3b5ac75, 0x8844f3f5, 0x58b0ce0d, 0xf9c6dd0c, 0x9afe47b4,
      0xc2468125, 0xa4ba16a1, 0x186171ec, 0x75a18672, 0x25e3bc04, 0xd4c524cc, 0x5b7729bb, 0x4298b3f4,
      0xd27a2f41, 0x9b967924, 0x39c11ad1, 0x8b724394, 0x88bc97a7, 0x2f49e3aa, 0xe30d7f1d, 0x0a90de92};
    static constexpr storage<limbs_count> m = {0xd59646e8, 0xec4f881f, 0x8163c701, 0x4e65c59e, 0x80a19de7, 0x2f7d1dc7,
                                               0x7fda82a5, 0xa46e09d0, 0x331e9ae8, 0x38a0406c, 0xcf327917, 0x2760d74b};
    static constexpr storage<limbs_count> one = {0x00000001, 0x00000000, 0x00000000, 0x00000000,
                                                 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                                 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> zero = {0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                                  0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                                  0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> montgomery_r = {0x0002fffd, 0x76090000, 0xc40c0002, 0xebf4000b,
                                                          0x53c758ba, 0x5f489857, 0x70525745, 0x77ce5853,
                                                          0xa256ec6d, 0x5c071a97, 0xfa80e493, 0x15f65ec3};
    static constexpr storage<limbs_count> montgomery_r_inv = {0x380b4820, 0xf4d38259, 0xd898fafb, 0x7fe11274,
                                                              0x14956dc8, 0x343ea979, 0x58a88de9, 0x1797ab14,
                                                              0x3c4f538b, 0xed5e6427, 0xe8fb0ce9, 0x14fec701};
    // nonresidue to generate the extension field
    static constexpr uint32_t nonresidue = 1;
    // true if nonresidue is negative
    static constexpr bool nonresidue_is_negative = true;
  };
} // namespace bls12_381

#endif
