#pragma once

#include "icicle/fields/snark_fields/bls12_377_base.h"
#include "icicle/math/storage.h"
#include "icicle/fields/field.h"
#include "icicle/fields/complex_extension.h"
#include "icicle/fields/cubic_extension.h"

namespace bls12_377 {
  typedef Field<fq_config> point_field_t;
  typedef ComplexExtensionField<fq_config, point_field_t> g2_point_field_t;

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
  typedef fq12_field_t target_field_t;
} // namespace bls12_377