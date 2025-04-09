#pragma once

#include "icicle/fields/snark_fields/bls12_381_base.h"
#include "icicle/math/storage.h"
#include "icicle/fields/field.h"
#include "icicle/fields/complex_extension.h"
#include "icicle/fields/cubic_extension.h"

namespace bls12_381 {
  typedef Field<fq_config> point_field_t;
  typedef ComplexExtensionField<fq_config, point_field_t> g2_point_field_t;

  static constexpr point_field_t BASE_FIELD_FROBENIUS_COEFF_C1[2] = {
    point_field_t::one(),
    {{0xffffaaaa, 0xb9feffff, 0xb153ffff, 0x1eabfffe, 0xf6b0f624, 0x6730d2a0, 0xf38512bf, 0x64774b84, 0x434bacd7,
      0x4b1ba7b6, 0x397fe69a, 0x1a0111ea}}};

  static void mul_fp2_field_by_frob_coeff(g2_point_field_t& fe, unsigned power)
  {
    fe.c1 = fe.c1 * BASE_FIELD_FROBENIUS_COEFF_C1[power % 2];
  }

  struct fq6_config {
    // nonresidue to generate the extension field
    static constexpr g2_point_field_t nonresidue = g2_point_field_t{point_field_t::one(), point_field_t::one()};
    // true if nonresidue is negative
    static constexpr bool nonresidue_is_negative = false;
    static constexpr bool nonresidue_is_u32 = false;

    static constexpr g2_point_field_t FROBENIUS_COEFF_C1[6] = {
      {point_field_t::one(), point_field_t::zero()},
      {point_field_t::zero(),
       {{0x0000aaac, 0x8bfd0000, 0x4f49fffd, 0x409427eb, 0x0fb85f9b, 0x897d2965, 0x89759ad4, 0xaa0d857d, 0x63d4de85,
         0xec024086, 0x397fe699, 0x1a0111ea}}},
      {{{0xfffefffe, 0x2e01ffff, 0x620a0002, 0xde17d813, 0xe6f89688, 0xddb3a93b, 0x6a0f77ea, 0xba69c607, 0xdf76ce51,
         0x5f19672f}},
       point_field_t::zero()},
      {point_field_t::zero(), point_field_t::one()},
      {{{0x0000aaac, 0x8bfd0000, 0x4f49fffd, 0x409427eb, 0x0fb85f9b, 0x897d2965, 0x89759ad4, 0xaa0d857d, 0x63d4de85,
         0xec024086, 0x397fe699, 0x1a0111ea}},
       point_field_t::zero()},
      {point_field_t::zero(),
       {{0xfffefffe, 0x2e01ffff, 0x620a0002, 0xde17d813, 0xe6f89688, 0xddb3a93b, 0x6a0f77ea, 0xba69c607, 0xdf76ce51,
         0x5f19672f}}}};

    static constexpr g2_point_field_t FROBENIUS_COEFF_C2[6] = {
      {point_field_t::one(), point_field_t::zero()},
      {{{0x0000aaad, 0x8bfd0000, 0x4f49fffd, 0x409427eb, 0x0fb85f9b, 0x897d2965, 0x89759ad4, 0xaa0d857d, 0x63d4de85,
         0xec024086, 0x397fe699, 0x1a0111ea}},
       point_field_t::zero()},
      {{{0x0000aaac, 0x8bfd0000, 0x4f49fffd, 0x409427eb, 0x0fb85f9b, 0x897d2965, 0x89759ad4, 0xaa0d857d, 0x63d4de85,
         0xec024086, 0x397fe699, 0x1a0111ea}},
       point_field_t::zero()

      },
      {{{0xffffaaaa, 0xb9feffff, 0xb153ffff, 0x1eabfffe, 0xf6b0f624, 0x6730d2a0, 0xf38512bf, 0x64774b84, 0x434bacd7,
         0x4b1ba7b6, 0x397fe69a, 0x1a0111ea}},
       point_field_t::zero()

      },
      {{{0xfffefffe, 0x2e01ffff, 0x620a0002, 0xde17d813, 0xe6f89688, 0xddb3a93b, 0x6a0f77ea, 0xba69c607, 0xdf76ce51,
         0x5f19672f}},
       point_field_t::zero()

      },
      {{{0xfffeffff, 0x2e01ffff, 0x620a0002, 0xde17d813, 0xe6f89688, 0xddb3a93b, 0x6a0f77ea, 0xba69c607, 0xdf76ce51,
         0x5f19672f}},
       point_field_t::zero()

      }};

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

  struct fq12_config {
    // nonresidue to generate the extension field
    static constexpr fq6_field_t nonresidue =
      fq6_field_t{g2_point_field_t::zero(), g2_point_field_t::one(), g2_point_field_t::zero()};
    // true if nonresidue is negative
    static constexpr bool nonresidue_is_negative = false;
    static constexpr bool nonresidue_is_u32 = false;

    static constexpr g2_point_field_t FROBENIUS_COEFF_C1[12] = {
      {
        point_field_t::one(),
        point_field_t::zero(),
      },
      {
        {{0x92235fb8, 0x8d0775ed, 0x63e7813d, 0xf67ea53d, 0x84bab9c4, 0x7b2443d7, 0x3cbd5f4f, 0x0fd603fd, 0x202c0d1f,
          0xc231beb4, 0x02bb0667, 0x1904d3bf}},
        {{0x6ddc4af3, 0x2cf78a12, 0x4d6c7ec2, 0x282d5ac1, 0x71f63c5f, 0xec0c8ec9, 0xb6c7b36f, 0x54a14787, 0x231f9fb8,
          0x88e9e902, 0x36c4e032, 0x00fc3e2b}},
      },
      {
        {{0xfffeffff, 0x2e01ffff, 0x620a0002, 0xde17d813, 0xe6f89688, 0xddb3a93b, 0x6a0f77ea, 0xba69c607, 0xdf76ce51,
          0x5f19672f}},
        point_field_t::zero(),
      },
      {
        {{0x121bdea2, 0xf1ee7b04, 0x3e67fa0a, 0x304466cf, 0xf61eb45e, 0xef396489, 0x30b1cf60, 0x1c3dedd9, 0xd77a2cd9,
          0xe2e9c448, 0x0180a68e, 0x135203e6}},
        {{0xede3cc09, 0xc81084fb, 0x72ec05f4, 0xee67992f, 0x009241c5, 0x77f76e17, 0xc2d3435e, 0x48395dab, 0x6bd17ffe,
          0x6831e36d, 0x37ff400b, 0x06af0e04}},
      },
      {
        {{0xfffefffe, 0x2e01ffff, 0x620a0002, 0xde17d813, 0xe6f89688, 0xddb3a93b, 0x6a0f77ea, 0xba69c607, 0xdf76ce51,
          0x5f19672f}},
        point_field_t::zero(),
      },
      {
        {{0x7ff82995, 0x1ee60516, 0x8bd478cd, 0x5871c190, 0x6814f0bd, 0xdb45f353, 0xe77982d0, 0x70df3560, 0xfa99cc91,
          0x6bd3ad4a, 0x384586c1, 0x144e4211}},
        {{0x80078116, 0x9b18fae9, 0x257f8732, 0xc63a3e6e, 0x8e9c0566, 0x8beadf4d, 0x0c0b8fee, 0xf3981624, 0x48b1e045,
          0xdf47fa6b, 0x013a5fd8, 0x05b2cfd9}},
      },
      {
        {{0xffffaaaa, 0xb9feffff, 0xb153ffff, 0x1eabfffe, 0xf6b0f624, 0x6730d2a0, 0xf38512bf, 0x64774b84, 0x434bacd7,
          0x4b1ba7b6, 0x397fe69a, 0x1a0111ea}},
        point_field_t::zero(),
      },
      {
        {{0x6ddc4af3, 0x2cf78a12, 0x4d6c7ec2, 0x282d5ac1, 0x71f63c5f, 0xec0c8ec9, 0xb6c7b36f, 0x54a14787, 0x231f9fb8,
          0x88e9e902, 0x36c4e032, 0x00fc3e2b}},
        {{0x92235fb8, 0x8d0775ed, 0x63e7813d, 0xf67ea53d, 0x84bab9c4, 0x7b2443d7, 0x3cbd5f4f, 0x0fd603fd, 0x202c0d1f,
          0xc231beb4, 0x02bb0667, 0x1904d3bf}},
      },
      {
        {{0x0000aaac, 0x8bfd0000, 0x4f49fffd, 0x409427eb, 0x0fb85f9b, 0x897d2965, 0x89759ad4, 0xaa0d857d, 0x63d4de85,
          0xec024086, 0x397fe699, 0x1a0111ea}},
        point_field_t::zero(),
      },
      {
        {{0xede3cc09, 0xc81084fb, 0x72ec05f4, 0xee67992f, 0x009241c5, 0x77f76e17, 0xc2d3435e, 0x48395dab, 0x6bd17ffe,
          0x6831e36d, 0x37ff400b, 0x06af0e04}},
        {{0x121bdea2, 0xf1ee7b04, 0x3e67fa0a, 0x304466cf, 0xf61eb45e, 0xef396489, 0x30b1cf60, 0x1c3dedd9, 0xd77a2cd9,
          0xe2e9c448, 0x0180a68e, 0x135203e6}},
      },
      {{{0x0000aaad, 0x8bfd0000, 0x4f49fffd, 0x409427eb, 0x0fb85f9b, 0x897d2965, 0x89759ad4, 0xaa0d857d, 0x63d4de85,
         0xec024086, 0x397fe699, 0x1a0111ea}},
       point_field_t::zero()},
      {{{0x80078116, 0x9b18fae9, 0x257f8732, 0xc63a3e6e, 0x8e9c0566, 0x8beadf4d, 0x0c0b8fee, 0xf3981624, 0x48b1e045,
         0xdf47fa6b, 0x013a5fd8, 0x05b2cfd9}},
       {{0x7ff82995, 0x1ee60516, 0x8bd478cd, 0x5871c190, 0x6814f0bd, 0xdb45f353, 0xe77982d0, 0x70df3560, 0xfa99cc91,
         0x6bd3ad4a, 0x384586c1, 0x144e4211}}}};
  };
  typedef ComplexExtensionField<fq12_config, fq6_field_t> fq12_field_t; // T3
  typedef fq12_field_t target_field_t;
} // namespace bls12_381