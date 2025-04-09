#pragma once

#include "icicle/fields/cubic_extension.h"
#include "icicle/curves/params/bn254.h"
#include "icicle/pairing/models/bn.h"

namespace pairing_bn254 {
  using namespace bn254;
  using namespace icicle_bn_pairing;

  struct PairingConfig {
    static constexpr storage<2> Z = {0x4a6909f1, 0x44e992b4};
    static constexpr bool Z_IS_NEGATIVE = false;

    static constexpr int ATE_LOOP_BITS[] = {0,  0, 0, 1,  0, 1, 0, -1, 0,  0, -1, 0,  0, 0,  1, 0, 0, -1, 0, -1, 0, 0,
                                            0,  1, 0, -1, 0, 0, 0, 0,  -1, 0, 0,  1,  0, -1, 0, 0, 1, 0,  0, 0,  0, 0,
                                            -1, 0, 0, -1, 0, 1, 0, -1, 0,  0, 0,  -1, 0, -1, 0, 0, 0, 1,  0, 1,  1};
    static constexpr int Z_NAF[] = {1,  0, 0, 0, -1, 0,  0, 0, 0, 1, 0, 1,  0, 0, 0, 0,  1, 0,  0, 1,  0,
                                    -1, 0, 1, 0, 1,  0,  1, 0, 0, 1, 0, 0,  0, 1, 0, -1, 0, -1, 0, -1, 0,
                                    1,  0, 1, 0, 0,  -1, 0, 1, 0, 1, 0, -1, 0, 0, 1, 0,  1, 0,  0, 0,  1};

    static constexpr TwistType TWIST_TYPE = TwistType::D;
    static constexpr g2_point_field_t TWIST_MUL_BY_Q_X = {
      {{0x176f553d, 0x99e39557, 0xc2c3330c, 0xb78cc310, 0xf559b143, 0x4c0bec3c, 0x4f7911f7, 0x2fb34798}},
      {{0x640fcba2, 0x1665d51c, 0x0b7c9dce, 0x32ae2a1d, 0xd75a0794, 0x4ba4cc8b, 0x61ebae20, 0x16c9e550}}};
    static constexpr g2_point_field_t TWIST_MUL_BY_Q_Y = {
      {{0x71a0135a, 0xdc540146, 0xa9c95998, 0xdbaae0ed, 0xb6e2f9b9, 0xdc5ec698, 0x489af5dc, 0x063cf305}},
      {{0x2623b0e3, 0x82d37f63, 0x8fa25bd2, 0x21807dc9, 0xec796f2b, 0x0704b5a7, 0xac41049a, 0x07c03cbc}}};

    static void mul_fp2_field_by_frob_coeff(g2_point_field_t& fe, unsigned power)
    {
      bn254::mul_fp2_field_by_frob_coeff(fe, power);
    }

    static constexpr g2_point_field_t CUBIC_NONRESIDUE = fq6_config::nonresidue;

    static void mul_fp2_by_nonresidue(g2_point_field_t& f)
    {
      // (9*c0-c1)+u*(9*c1+c0)
      g2_point_field_t f8 = f * point_field_t::from(8);
      point_field_t c0 = f.c1;
      c0 = point_field_t::neg(c0); // mul fp by nonresidue (-1)
      c0 = c0 + f8.c0 + f.c0;
      point_field_t c1 = f8.c1 + f.c1 + f.c0;
      f = g2_point_field_t{c0, c1};
    }

    static void mul_fp6_by_nonresidue(fq6_field_t& f)
    {
      g2_point_field_t t = f.c1;
      f.c1 = f.c0;
      f.c0 = f.c2;
      mul_fp2_by_nonresidue(f.c0);
      f.c2 = t;
    }

    static void frobenius_map(fq12_field_t& f, unsigned power)
    {
      fq6_config::frobenius_map(f.c0.c0, f.c0.c1, f.c0.c2, power);
      fq6_config::frobenius_map(f.c1.c0, f.c1.c1, f.c1.c2, power);
      f.c1 *= fq12_config::FROBENIUS_COEFF_C1[power % 12];
    }

    using Fp = point_field_t;             // Base prime field
    using Fp2 = g2_point_field_t;         // Quadratic extension
    using Fp6 = fq6_field_t;              // Cubic extension over Fp2
    using Fp12 = fq12_field_t;            // Quadratic extension over Fp6
    using G1Affine = affine_t;            // G1 group (affine coordinates)
    using G1Projective = projective_t;    // G1 group (projective coordinates)
    using G2Affine = g2_affine_t;         // G2 group (affine coordinates)
    using G2Projective = g2_projective_t; // G2 group (projective coordinates)
    using TargetField = Fp12;             // Result of the pairing
    using G2Config = G2;
  };

  // Alias the pairing methods
  using icicle_bn_pairing::final_exponentiation;
  using icicle_bn_pairing::miller_loop;
}; // namespace pairing_bn254