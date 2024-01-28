#pragma once
#ifndef GRUMPKIN_PARAMS_H
#define GRUMPKIN_PARAMS_H

#include "../utils/storage.cuh"

namespace grumpkin {
  struct fp_config {
    static constexpr unsigned limbs_count = 8;
    static constexpr unsigned omegas_count = 28;
    static constexpr unsigned modulus_bit_count = 254;

    static constexpr unsigned num_of_reductions = 1;
    // neg_modulus should be changed
    static constexpr storage<limbs_count> neg_modulus = {0x278302b9, 0xc3df73e9, 0x978e3572, 0x687e956e,
                                                         0x7e7ea7a2, 0x47afba49, 0x1ece5fd6, 0xcf9bb18d};
    static constexpr storage<limbs_count> modulus = {0xd87cfd47, 0x3c208c16, 0x6871ca8d, 0x97816a91, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
    static constexpr storage<limbs_count> modulus_2 = {0xb0f9fa8e, 0x7841182d, 0xd0e3951a, 0x2f02d522, 0x0302b0bb, 0x70a08b6d, 0xc2634053, 0x60c89ce5};
    static constexpr storage<limbs_count> modulus_4 = {0x61f3f51c, 0xf082305b, 0xa1c72a34, 0x5e05aa45, 0x06056176, 0xe14116da, 0x84c680a6, 0xc19139cb};
    static constexpr storage<2*limbs_count> modulus_wide = {0xd87cfd47, 0x3c208c16, 0x6871ca8d, 0x97816a91, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<2*limbs_count> modulus_squared = {0x275d69b1, 0x3b5458a2, 0x09eac101, 0xa602072d, 0x6d96cadc, 0x4a50189c, 0x7a1242c8, 0x04689e95, 0x34c6b38d, 0x26edfa5c, 0x16375606, 0xb00b8551, 0x0348d21c, 0x599a6f7c, 0x763cbf9c, 0x0925c4b8};
    static constexpr storage<2*limbs_count> modulus_squared_2 = {0x4ebad362, 0x76a8b144, 0x13d58202, 0x4c040e5a, 0xdb2d95b9, 0x94a03138, 0xf4248590, 0x08d13d2a, 0x698d671a, 0x4ddbf4b8, 0x2c6eac0c, 0x60170aa2, 0x0691a439, 0xb334def8, 0xec797f38, 0x124b8970};
    static constexpr storage<2*limbs_count> modulus_squared_4 = {0x9d75a6c4, 0xed516288, 0x27ab0404, 0x98081cb4, 0xb65b2b72, 0x29406271, 0xe8490b21, 0x11a27a55, 0xd31ace34, 0x9bb7e970, 0x58dd5818, 0xc02e1544, 0x0d234872, 0x6669bdf0, 0xd8f2fe71, 0x249712e1};

    static constexpr storage<limbs_count> m = {0x19bf90e5, 0x6f3aed8a, 0x67cd4c08, 0xae965e17, 0x68073013, 0xab074a58, 0x623a04a7, 0x54a47462};
    static constexpr storage<limbs_count> one = {0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> zero = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> montgomery_r = {0xc58f0d9d, 0xd35d438d, 0xf5c70b3d, 0x0a78eb28, 0x7879462c, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1};
    static constexpr storage<limbs_count> montgomery_r_inv = {0x014afa37, 0xed84884a, 0x0278edf8, 0xeb202285, 0xb74492d9, 0xcf63e9cf, 0x59e5c639, 0x2e671571};
  };

  struct fq_config {
    static constexpr unsigned limbs_count = 8;
    static constexpr unsigned modulus_bit_count = 254;
    static constexpr unsigned num_of_reductions = 1;
    // neg_modulus should be changed
    static constexpr storage<limbs_count> modulus = {0xf0000001, 0x43e1f593, 0x79b97091, 0x2833e848, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
    static constexpr storage<limbs_count> modulus_2 = {0xe0000002, 0x87c3eb27, 0xf372e122, 0x5067d090, 0x0302b0ba, 0x70a08b6d, 0xc2634053, 0x60c89ce5};
    static constexpr storage<limbs_count> modulus_4 = {0xc0000004, 0x0f87d64f, 0xe6e5c245, 0xa0cfa121, 0x06056174, 0xe14116da, 0x84c680a6, 0xc19139cb};
    static constexpr storage<limbs_count> neg_modulus = {0x0fffffff, 0xbc1e0a6c, 0x86468f6e, 0xd7cc17b7,
                                                         0x7e7ea7a2, 0x47afba49, 0x1ece5fd6, 0xcf9bb18d};
    static constexpr storage<2*limbs_count> modulus_wide = {0xf0000001, 0x43e1f593, 0x79b97091, 0x2833e848, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<2*limbs_count> modulus_squared = {0xe0000001, 0x08c3eb27, 0xdcb34000, 0xc7f26223, 0x68c9bb7f, 0xffe9a62c, 0xe821ddb0, 0xa6ce1975, 0x47b62fe7, 0x2c77527b, 0xd379d3df, 0x85f73bb0, 0x0348d21c, 0x599a6f7c, 0x763cbf9c, 0x0925c4b8};
    static constexpr storage<2*limbs_count> modulus_squared_2 = {0xc0000002, 0x1187d64f, 0xb9668000, 0x8fe4c447, 0xd19376ff, 0xffd34c58, 0xd043bb61, 0x4d9c32eb, 0x8f6c5fcf, 0x58eea4f6, 0xa6f3a7be, 0x0bee7761, 0x0691a439, 0xb334def8, 0xec797f38, 0x124b8970};
    static constexpr storage<2*limbs_count> modulus_squared_4 = {0x80000004, 0x230fac9f, 0x72cd0000, 0x1fc9888f, 0xa326edff, 0xffa698b1, 0xa08776c3, 0x9b3865d7, 0x1ed8bf9e, 0xb1dd49ed, 0x4de74f7c, 0x17dceec3, 0x0d234872, 0x6669bdf0, 0xd8f2fe71, 0x249712e1};
    static constexpr storage<limbs_count> m = {0xbe1de925, 0x620703a6, 0x09e880ae, 0x71448520, 0x68073014, 0xab074a58, 0x623a04a7, 0x54a47462};
    static constexpr storage<limbs_count> one = {0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> zero = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> montgomery_r = {0x4ffffffb, 0xac96341c, 0x9f60cd29, 0x36fc7695, 0x7879462e, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1};
    static constexpr storage<limbs_count> montgomery_r_inv = {0x6db1194e, 0xdc5ba005, 0xe111ec87, 0x090ef5a9, 0xaeb85d5d, 0xc8260de4, 0x82c5551c, 0x15ebf951};
    // i^2, the square of the imaginary unit for the extension field
    static constexpr uint32_t i_squared = 1;
    // true if i^2 is negative
    static constexpr bool i_squared_is_negative = true;
  };

  // G1 and G2 generators
  static constexpr storage<fq_config::limbs_count> g1_gen_x = {0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
  static constexpr storage<fq_config::limbs_count> g1_gen_y = {0x448c41d8, 0x11b2dff1, 0x21c77dc3, 0x23d3446f, 0x35dfafbb, 0xaa7b8cf4, 0x9dc25d68, 0x14b34cf6};
  static constexpr storage<fq_config::limbs_count> g2_gen_x_re = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
  static constexpr storage<fq_config::limbs_count> g2_gen_x_im = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
  static constexpr storage<fq_config::limbs_count> g2_gen_y_re = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
  static constexpr storage<fq_config::limbs_count> g2_gen_y_im = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};

  static constexpr storage<fq_config::limbs_count> weierstrass_b = {0x6000005a, 0xdd705602, 0xcb319311, 0x223fa97a, 0x877910c0, 0xcc388229, 0x2b724eaa, 0x03439463};
  static constexpr storage<fq_config::limbs_count> weierstrass_b_g2_re = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
  static constexpr storage<fq_config::limbs_count> weierstrass_b_g2_im = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
}

#endif
