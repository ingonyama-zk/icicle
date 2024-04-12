#pragma once
#ifndef STARK_PARAMS_H
#define STARK_PARAMS_H

#include "utils/storage.cuh"

namespace stark {
  struct fp_config {
    static constexpr unsigned limbs_count = 8;
    static constexpr unsigned omegas_count = 192;
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

    static constexpr storage_array<omegas_count, limbs_count> omega = {
      {{0x42f8ef94, 0x6070024f, 0xe11a6161, 0xad187148, 0x9c8b0fa5, 0x3f046451, 0x87529cfa, 0x005282db},
       }};

    static constexpr storage_array<omegas_count, limbs_count> omega_inv = {
      {{0xf0000000, 0x43e1f593, 0x79b97091, 0x2833e848, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72},
       }};

    static constexpr storage_array<omegas_count, limbs_count> inv = {
      {{0xf8000001, 0xa1f0fac9, 0x3cdcb848, 0x9419f424, 0x40c0ac2e, 0xdc2822db, 0x7098d014, 0x18322739},
       }};
  };

// Fp in Gnark
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
    // i^2, the square of the imaginary unit for the extension field
    static constexpr uint32_t i_squared = 1;
    // true if i^2 is negative
    static constexpr bool i_squared_is_negative = true;
  };

  // G1 generators
  static constexpr storage<fq_config::limbs_count> g1_gen_x = {0xc943cfca, 0x3d723d8b, 0x0d1819e0, 0xdeacfd9b, 0x5a40f0c7, 0x7beced41, 0x8599971b, 0x01ef15c1};
  static constexpr storage<fq_config::limbs_count> g1_gen_y = {0x36e8dc1f, 0x2873000c, 0x1abe43a3, 0xde53ecd1, 0xdf46ec62, 0xb7be4801, 0x0aa49730, 0x00566806};

  static constexpr storage<fq_config::limbs_count> weierstrass_b = {0x9cee9e89, 0xf4cdfcb9, 0x15c915c1, 0x609ad26c, 0x72f7a8c5, 0x150e596d, 0xefbe40de, 0x06f21413};
} // namespace bn254

#endif
