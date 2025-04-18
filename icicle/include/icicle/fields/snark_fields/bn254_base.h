#pragma once

#include "icicle/math/storage.h"
#include "icicle/fields/params_gen.h"

namespace bn254 {
  struct fq_config {
    static constexpr storage<8> modulus = {0xd87cfd47, 0x3c208c16, 0x6871ca8d, 0x97816a91,
                                           0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
    static constexpr unsigned reduced_digits_count = 3;
    static constexpr storage_array<reduced_digits_count, 8> reduced_digits = {
      {{0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
       {0xc58f0d9d, 0xd35d438d, 0xf5c70b3d, 0x0a78eb28, 0x7879462c, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1},
       {0x538afa89, 0xf32cfc5b, 0xd44501fb, 0xb5e71911, 0x0a417ff6, 0x47ab1eff, 0xcab8351f, 0x06d89f71}}};
    PARAMS(modulus)
    MOD_SQR_SUBS()
    static constexpr storage_array<mod_subs_count, 2 * limbs_count + 2> mod_subs = {
      {{0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
       {0x0261419e, 0x18754db1, 0x31e16455, 0x91d4069a, 0x12e13d0e, 0xdbd801ff, 0x6a07e140, 0xe613f157, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x07ffffff, 0x00000000, 0x00000000},
       {0xdd3f8083, 0x6d0b2778, 0xcc349337, 0xbb2977c5, 0xa743d27a, 0x700049b4, 0xb54162ab, 0xfc8c3121, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x0fffffff, 0x00000000, 0x00000000},
       {0xdfa0c221, 0x85807529, 0xfe15f78c, 0x4cfd7e5f, 0xba250f89, 0x4bd84bb3, 0x1f4943ec, 0xe2a02279, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x17ffffff, 0x00000000, 0x00000000},
       {0xba7f0106, 0xda164ef1, 0x9869266e, 0x7652ef8b, 0x4e87a4f5, 0xe0009369, 0x6a82c556, 0xf9186243, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x1fffffff, 0x00000000, 0x00000000},
       {0xbce042a4, 0xf28b9ca2, 0xca4a8ac3, 0x0826f625, 0x6168e204, 0xbbd89568, 0xd48aa697, 0xdf2c539a, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x27ffffff, 0x00000000, 0x00000000},
       {0x97be8189, 0x4721766a, 0x649db9a6, 0x317c6751, 0xf5cb7770, 0x5000dd1d, 0x1fc42802, 0xf5a49365, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x2fffffff, 0x00000000, 0x00000000},
       {0x9a1fc327, 0x5f96c41b, 0x967f1dfb, 0xc3506deb, 0x08acb47e, 0x2bd8df1d, 0x89cc0943, 0xdbb884bc, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x37ffffff, 0x00000000, 0x00000000},
       {0x74fe020c, 0xb42c9de3, 0x30d24cdd, 0xeca5df17, 0x9d0f49ea, 0xc00126d2, 0xd5058aad, 0xf230c486, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x3fffffff, 0x00000000, 0x00000000},
       {0x775f43aa, 0xcca1eb94, 0x62b3b132, 0x7e79e5b1, 0xaff086f9, 0x9bd928d1, 0x3f0d6bee, 0xd844b5de, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x47ffffff, 0x00000000, 0x00000000},
       {0x523d828f, 0x2137c55c, 0xfd06e015, 0xa7cf56dc, 0x44531c65, 0x30017087, 0x8a46ed59, 0xeebcf5a8, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x4fffffff, 0x00000000, 0x00000000},
       {0x549ec42d, 0x39ad130d, 0x2ee8446a, 0x39a35d77, 0x57345974, 0x0bd97286, 0xf44ece9a, 0xd4d0e6ff, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x57ffffff, 0x00000000, 0x00000000},
       {0x2f7d0312, 0x8e42ecd5, 0xc93b734c, 0x62f8cea2, 0xeb96eee0, 0xa001ba3b, 0x3f885004, 0xeb4926ca, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x5fffffff, 0x00000000, 0x00000000},
       {0x31de44b0, 0xa6b83a86, 0xfb1cd7a1, 0xf4ccd53c, 0xfe782bee, 0x7bd9bc3a, 0xa9903145, 0xd15d1821, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x67ffffff, 0x00000000, 0x00000000},
       {0x0cbc8395, 0xfb4e144e, 0x95700683, 0x1e224668, 0x92dac15b, 0x100203f0, 0xf4c9b2b0, 0xe7d557eb, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x6fffffff, 0x00000000, 0x00000000},
       {0xe79ac27a, 0x4fe3ee15, 0x2fc33566, 0x4777b794, 0x273d56c7, 0xa42a4ba6, 0x4003341a, 0xfe4d97b6, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x77ffffff, 0x00000000, 0x00000000},
       {0xe9fc0418, 0x68593bc6, 0x61a499bb, 0xd94bbe2e, 0x3a1e93d5, 0x80024da5, 0xaa0b155b, 0xe461890d, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x7fffffff, 0x00000000, 0x00000000},
       {0xc4da42fd, 0xbcef158e, 0xfbf7c89d, 0x02a12f59, 0xce812942, 0x142a955a, 0xf54496c6, 0xfad9c8d7, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x87ffffff, 0x00000000, 0x00000000},
       {0xc73b849b, 0xd564633f, 0x2dd92cf2, 0x947535f4, 0xe1626650, 0xf0029759, 0x5f4c7806, 0xe0edba2f, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x8fffffff, 0x00000000, 0x00000000},
       {0xa219c380, 0x29fa3d07, 0xc82c5bd5, 0xbdcaa71f, 0x75c4fbbc, 0x842adf0f, 0xaa85f971, 0xf765f9f9, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x97ffffff, 0x00000000, 0x00000000},
       {0xa47b051e, 0x426f8ab8, 0xfa0dc02a, 0x4f9eadb9, 0x88a638cb, 0x6002e10e, 0x148ddab2, 0xdd79eb51, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x9fffffff, 0x00000000, 0x00000000},
       {0x7f594403, 0x97056480, 0x9460ef0c, 0x78f41ee5, 0x1d08ce37, 0xf42b28c4, 0x5fc75c1c, 0xf3f22b1b, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xa7ffffff, 0x00000000, 0x00000000},
       {0x81ba85a1, 0xaf7ab231, 0xc6425361, 0x0ac8257f, 0x2fea0b46, 0xd0032ac3, 0xc9cf3d5d, 0xda061c72, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xafffffff, 0x00000000, 0x00000000},
       {0x5c98c486, 0x04108bf9, 0x60958244, 0x341d96ab, 0xc44ca0b2, 0x642b7278, 0x1508bec8, 0xf07e5c3d, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xb7ffffff, 0x00000000, 0x00000000}}};

    // nonresidue to generate the extension field
    static constexpr uint32_t nonresidue = 1;
    // true if nonresidue is negative
    static constexpr bool nonresidue_is_negative = true;
    static constexpr bool nonresidue_is_u32 = true;
  };
} // namespace bn254
