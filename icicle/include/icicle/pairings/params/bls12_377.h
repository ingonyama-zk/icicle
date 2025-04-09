#pragma once

#include "icicle/curves/params/bls12_377.h"
#include "icicle/pairings/models/bls12.h"

namespace pairing_bls12_377 {
  using namespace bls12_377;
  using namespace icicle_bls12_pairing;

  struct PairingConfig {
    static constexpr storage<2> Z = {0x00000001, 0x8508c000};
    static constexpr bool Z_IS_NEGATIVE = false;
    static constexpr int Z_NAF[] = {1, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, -1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1};

    static constexpr TwistType TWIST_TYPE = TwistType::D;

    static constexpr point_field_t BASE_FIELD_FROBENIUS_COEFF_C1[2] = {
      point_field_t::one(),
      {{0x00000000, 0x8508c000, 0x30000000, 0x170b5d44, 0xba094800, 0x1ef3622f, 0x00f5138f, 0x1a22d9f3, 0x6ca1493b,
        0xc63b05c0, 0x17c510ea, 0x01ae3a46}}};

    static void mul_fp2_field_by_frob_coeff(g2_point_field_t& fe, unsigned power)
    {
      fe.c1 = fe.c1 * BASE_FIELD_FROBENIUS_COEFF_C1[power % 2];
    }

    struct fq6_config {
      // nonresidue to generate the extension field
      static constexpr g2_point_field_t nonresidue = g2_point_field_t{point_field_t::zero(), point_field_t::one()};
      // true if nonresidue is negative
      static constexpr bool nonresidue_is_negative = false;
      static constexpr bool nonresidue_is_u32 = false;

      static constexpr g2_point_field_t FROBENIUS_COEFF_C1[3] = {
        {point_field_t::one(), point_field_t::zero()},
        {{{0x00000002, 0x8508c000, 0x90000000, 0x452217cc, 0x970dec00, 0xc5ed1347, 0x34594aab, 0x619aaf7d, 0xdd14f6ec,
           0x09b3af05}},
         point_field_t::zero()},
        {{{0x00000001, 0x8508c000, 0x90000000, 0x452217cc, 0x970dec00, 0xc5ed1347, 0x34594aab, 0x619aaf7d, 0xdd14f6ec,
           0x09b3af05}},
         point_field_t::zero()}};

      static constexpr g2_point_field_t FROBENIUS_COEFF_C2[3] = {
        {point_field_t::one(), point_field_t::zero()},
        {{{0x00000001, 0x8508c000, 0x90000000, 0x452217cc, 0x970dec00, 0xc5ed1347, 0x34594aab, 0x619aaf7d, 0xdd14f6ec,
           0x09b3af05}},
         point_field_t::zero()},
        {{{0xffffffff, 0xffffffff, 0x9fffffff, 0xd1e94577, 0x22fb5bff, 0x59064ee8, 0xcc9bc8e3, 0xb8882a75, 0x8f8c524e,
           0xbc8756ba, 0x17c510ea, 0x01ae3a46}},
         point_field_t::zero()}};

      static void frobenius_map(g2_point_field_t& c0, g2_point_field_t& c1, g2_point_field_t& c2, unsigned power)
      {
        mul_fp2_field_by_frob_coeff(c0, power);
        mul_fp2_field_by_frob_coeff(c1, power);
        mul_fp2_field_by_frob_coeff(c2, power);

        c1 *= FROBENIUS_COEFF_C1[power % 6];
        c2 *= FROBENIUS_COEFF_C2[power % 6];
      }
    };
    typedef CubicExtensionField<fq6_config, g2_point_field_t> fq6_field_t; // T2
    static constexpr g2_point_field_t CUBIC_NONRESIDUE = fq6_config::nonresidue;

    static void mul_fp2_by_nonresidue(g2_point_field_t& f)
    {
      point_field_t t = f.c0;
      f.c0 = f.c1;
      f.c0 = point_field_t::neg(f.c0);
      f.c0 = f.c0 + f.c0 * point_field_t::from(4);
      f.c1 = t;
    }

    static void mul_fp6_by_nonresidue(fq6_field_t& f)
    {
      g2_point_field_t t = f.c1;
      f.c1 = f.c0;
      f.c0 = f.c2;
      mul_fp2_by_nonresidue(f.c0);
      f.c2 = t;
    }

    struct fq12_config {
      // nonresidue to generate the extension field
      static constexpr fq6_field_t nonresidue =
        fq6_field_t{g2_point_field_t::zero(), g2_point_field_t::one(), g2_point_field_t::zero()};
      // true if nonresidue is negative
      static constexpr bool nonresidue_is_negative = false;
      static constexpr bool nonresidue_is_u32 = false;

      static constexpr g2_point_field_t FROBENIUS_COEFF_C1[3] = {
        {
          point_field_t::one(),
          point_field_t::zero(),
        },
        {{{0x104f2031, 0xe938a9d1, 0x58eb0188, 0xb57668e5, 0xa3aa559d, 0xc681bf34, 0xf94ebc8e, 0x5c8a45e0, 0x82567f91,
           0x33c1e306, 0x399c0196, 0x009a9975}},
         point_field_t::zero()},
        {
          {{0x00000002, 0x8508c000, 0x90000000, 0x452217cc, 0x970dec00, 0xc5ed1347, 0x34594aab, 0x619aaf7d, 0xdd14f6ec,
            0x09b3af05}},
          point_field_t::zero(),
        }};
    };
    typedef ComplexExtensionField<fq12_config, fq6_field_t> fq12_field_t; // T3

    static void frobenius_map(fq12_field_t& f, unsigned power)
    {
      fq6_config::frobenius_map(f.c0.c0, f.c0.c1, f.c0.c2, power);
      fq6_config::frobenius_map(f.c1.c0, f.c1.c1, f.c1.c2, power);
      f.c1 *= fq12_config::FROBENIUS_COEFF_C1[power % 12];
    }

    using Fp = point_field_t;     // Base prime field
    using Fp2 = g2_point_field_t; // Quadratic extension
    using Fp6 = fq6_field_t;      // Cubic extension over Fp2
    using Fp12 = fq12_field_t;    // Quadratic extension over Fp6
    using G1Affine = affine_t;    // G1 group (affine coordinates)
    using G2Affine = g2_affine_t; // G2 group (affine coordinates)
    using TargetField = Fp12;     // Result of the pairing
    using G2Config = G2;
  };

  // Alias the pairing methods
  using icicle_bls12_pairing::final_exponentiation;
  using icicle_bls12_pairing::miller_loop;
}; // namespace pairing_bls12_377