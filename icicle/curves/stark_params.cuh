#pragma once
#ifndef STARK_PARAMS_H
#define STARK_PARAMS_H

#include "utils/storage.cuh"

namespace stark {
  // Scalar Field Config
  // modulus = 3618502788666131213697322783095070105526743751716087489154079457884512865583
  struct fp_config {
    static constexpr unsigned limbs_count = 8;
    static constexpr unsigned omegas_count = 1;
    static constexpr unsigned modulus_bit_count = 252;
    static constexpr unsigned num_of_reductions = 1;

    static constexpr storage<limbs_count> modulus = {0xadc64d2f, 0x1e66a241, 0xcae7b232, 0xb781126d, 0xffffffff, 0xffffffff, 0x00000010, 0x08000000};
    static constexpr storage<limbs_count> modulus_2 = {0x5b8c9a5e, 0x3ccd4483, 0x95cf6464, 0x6f0224db, 0xffffffff, 0xffffffff, 0x00000021, 0x10000000};
    static constexpr storage<limbs_count> modulus_4 = {0xb71934bc, 0x799a8906, 0x2b9ec8c8, 0xde0449b7, 0xfffffffe, 0xffffffff, 0x00000043, 0x20000000};
    static constexpr storage<limbs_count> neg_modulus = {0x5239b2d1, 0xe1995dbe, 0x35184dcd, 0x487eed92, 0x00000000, 0x00000000, 0xffffffef, 0xf7ffffff};
    static constexpr storage<2 * limbs_count> modulus_wide = {0xadc64d2f, 0x1e66a241, 0xcae7b232, 0xb781126d, 0xffffffff, 0xffffffff, 0x00000010, 0x08000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<2 * limbs_count> modulus_squared = {0x01f94ea1, 0x33cc4bcb, 0x3385a741, 0xef31d230, 0x8dc40391, 0x005b5cc4, 0x0a9839be, 0x0e29314a, 0x0da20f7b, 0x810adcb9, 0xdcae7b19, 0xfb781126, 0x00000120, 0x10000000, 0x00000001, 0x00400000};
    static constexpr storage<2 * limbs_count> modulus_squared_2 = {0x03f29d42, 0x67989796, 0x670b4e82, 0xde63a460, 0x1b880723, 0x00b6b989, 0x1530737c, 0x1c526294, 0x1b441ef6, 0x0215b972, 0xb95cf633, 0xf6f0224d, 0x00000241, 0x20000000, 0x00000002, 0x00800000};
    static constexpr storage<2 * limbs_count> modulus_squared_4 = {0x07e53a84, 0xcf312f2c, 0xce169d04, 0xbcc748c0, 0x37100e47, 0x016d7312, 0x2a60e6f8, 0x38a4c528, 0x36883dec, 0x042b72e4, 0x72b9ec66, 0xede0449b, 0x00000483, 0x40000000, 0x00000004, 0x01000000};
    static constexpr storage<limbs_count> m = {0x384d77b0, 0x189ec175, 0xd32e2267, 0x21fbb648, 0x00009081, 0x00000000, 0xffffffbc, 0x1fffffff};
    static constexpr storage<limbs_count> one = {0x00000001, 0x00000000, 0x00000000, 0x00000000,
                                                 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> zero = {0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                                  0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> montgomery_r = {0xf4fca74f, 0x51925a0b, 0x6df16bee, 0xc75ec4b4, 0x00000008, 0x00000000, 0xfffffdf1, 0x07ffffff};
    static constexpr storage<limbs_count> montgomery_r_inv = {0x8b91307b, 0x81f69041, 0x65485708, 0xc0708974, 0x090c0159, 0xa541174a, 0xd6190374, 0x012bead2};

    static constexpr storage_array<omegas_count, limbs_count> omega = {
      {{0xadc64d2e, 0x1e66a241, 0xcae7b232, 0xb781126d, 0xffffffff, 0xffffffff, 0x00000010, 0x08000000},
       }};

    static constexpr storage_array<omegas_count, limbs_count> inv = {
      {{0xd6e32698, 0x0f335120, 0xe573d919, 0xdbc08936, 0xffffffff, 0x7fffffff, 0x00000008, 0x04000000},
       }};
  };

// BaseField Config
// modulus = 3618502788666131213697322783095070105623107215331596699973092056135872020481 (2^251+17*2^192+1)
  struct fq_config {
    static constexpr unsigned limbs_count = 8;
    static constexpr unsigned modulus_bit_count = 252;
    static constexpr unsigned num_of_reductions = 1;
    static constexpr storage<limbs_count> modulus = {0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000011, 0x08000000};
    static constexpr storage<limbs_count> modulus_2 = {0x00000002, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000022, 0x10000000};
    static constexpr storage<limbs_count> modulus_4 = {0x00000004, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000044, 0x20000000};
    static constexpr storage<limbs_count> neg_modulus = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffee, 0xf7ffffff};
    static constexpr storage<2 * limbs_count> modulus_wide = {
      0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000011, 0x08000000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<2 * limbs_count> modulus_squared = {0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000022, 0x10000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000121, 0x10000000, 0x00000001, 0x00400000};
    static constexpr storage<2 * limbs_count> modulus_squared_2 = {0x00000002, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000044, 0x20000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000242, 0x20000000, 0x00000002, 0x00800000};
    static constexpr storage<2 * limbs_count> modulus_squared_4 = {0x00000004, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000088, 0x40000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000484, 0x40000000, 0x00000004, 0x01000000};
    static constexpr storage<limbs_count> m = {0x8c81fffb, 0x00000002, 0xfeccf000, 0xffffffff, 0x0000907f, 0x00000000, 0xffffffbc, 0x1fffffff};
    static constexpr storage<limbs_count> one = {0x00000001, 0x00000000, 0x00000000, 0x00000000,
                                                 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> zero = {0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                                  0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> montgomery_r = {0xffffffe1, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xfffffdf0, 0x07ffffff};
    static constexpr storage<limbs_count> montgomery_r_inv = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000121, 0x10000000, 0x00000001, 0x00400000};
  };

  // G1 generators
  static constexpr storage<fq_config::limbs_count> g1_gen_x = {0xc943cfca, 0x3d723d8b, 0x0d1819e0, 0xdeacfd9b, 0x5a40f0c7, 0x7beced41, 0x8599971b, 0x01ef15c1};
  static constexpr storage<fq_config::limbs_count> g1_gen_y = {0x36e8dc1f, 0x2873000c, 0x1abe43a3, 0xde53ecd1, 0xdf46ec62, 0xb7be4801, 0x0aa49730, 0x00566806};

  static constexpr storage<fq_config::limbs_count> weierstrass_b = {0x9cee9e89, 0xf4cdfcb9, 0x15c915c1, 0x609ad26c, 0x72f7a8c5, 0x150e596d, 0xefbe40de, 0x06f21413};
} // namespace bn254

#endif
