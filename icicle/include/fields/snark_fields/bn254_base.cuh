#pragma once
#ifndef BN254_BASE_PARAMS_H
#define BN254_BASE_PARAMS_H

#include "../storage.cuh"

namespace bn254 {
  struct fq_config {
    static constexpr unsigned limbs_count = 8;
    static constexpr unsigned modulus_bit_count = 254;
    static constexpr unsigned num_of_reductions = 1;
    static constexpr storage<limbs_count> modulus = {0xd87cfd47, 0x3c208c16, 0x6871ca8d, 0x97816a91,
                                                     0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
    static constexpr storage<limbs_count> modulus_2 = {0xb0f9fa8e, 0x7841182d, 0xd0e3951a, 0x2f02d522,
                                                       0x0302b0bb, 0x70a08b6d, 0xc2634053, 0x60c89ce5};
    static constexpr storage<limbs_count> modulus_4 = {0x61f3f51c, 0xf082305b, 0xa1c72a34, 0x5e05aa45,
                                                       0x06056176, 0xe14116da, 0x84c680a6, 0xc19139cb};
    static constexpr storage<limbs_count> neg_modulus = {0x278302b9, 0xc3df73e9, 0x978e3572, 0x687e956e,
                                                         0x7e7ea7a2, 0x47afba49, 0x1ece5fd6, 0xcf9bb18d};
    static constexpr storage<2 * limbs_count> modulus_wide = {
      0xd87cfd47, 0x3c208c16, 0x6871ca8d, 0x97816a91, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<2 * limbs_count> modulus_squared = {
      0x275d69b1, 0x3b5458a2, 0x09eac101, 0xa602072d, 0x6d96cadc, 0x4a50189c, 0x7a1242c8, 0x04689e95,
      0x34c6b38d, 0x26edfa5c, 0x16375606, 0xb00b8551, 0x0348d21c, 0x599a6f7c, 0x763cbf9c, 0x0925c4b8};
    static constexpr storage<2 * limbs_count> modulus_squared_2 = {
      0x4ebad362, 0x76a8b144, 0x13d58202, 0x4c040e5a, 0xdb2d95b9, 0x94a03138, 0xf4248590, 0x08d13d2a,
      0x698d671a, 0x4ddbf4b8, 0x2c6eac0c, 0x60170aa2, 0x0691a439, 0xb334def8, 0xec797f38, 0x124b8970};
    static constexpr storage<2 * limbs_count> modulus_squared_4 = {
      0x9d75a6c4, 0xed516288, 0x27ab0404, 0x98081cb4, 0xb65b2b72, 0x29406271, 0xe8490b21, 0x11a27a55,
      0xd31ace34, 0x9bb7e970, 0x58dd5818, 0xc02e1544, 0x0d234872, 0x6669bdf0, 0xd8f2fe71, 0x249712e1};
    static constexpr storage<limbs_count> m = {0x19bf90e5, 0x6f3aed8a, 0x67cd4c08, 0xae965e17,
                                               0x68073013, 0xab074a58, 0x623a04a7, 0x54a47462};
    static constexpr storage<limbs_count> one = {0x00000001, 0x00000000, 0x00000000, 0x00000000,
                                                 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> zero = {0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                                  0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> montgomery_r = {0xc58f0d9d, 0xd35d438d, 0xf5c70b3d, 0x0a78eb28,
                                                          0x7879462c, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1};
    static constexpr storage<limbs_count> montgomery_r_inv = {0x014afa37, 0xed84884a, 0x0278edf8, 0xeb202285,
                                                              0xb74492d9, 0xcf63e9cf, 0x59e5c639, 0x2e671571};
    // nonresidue to generate the extension field
    static constexpr uint32_t nonresidue = 1;
    // true if nonresidue is negative
    static constexpr bool nonresidue_is_negative = true;
  };
} // namespace bn254

#endif
