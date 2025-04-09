#pragma once

#include "icicle/fields/snark_fields/bn254_base.h"
#include "icicle/math/storage.h"
#include "icicle/fields/field.h"
#include "icicle/fields/complex_extension.h"
#include "icicle/fields/cubic_extension.h"

namespace bn254 {
  typedef Field<fq_config> point_field_t;
  typedef ComplexExtensionField<fq_config, point_field_t> g2_point_field_t;

  static constexpr point_field_t BASE_FIELD_FROBENIUS_COEFF_C1[2] = {
    point_field_t::one(),
    {{0xd87cfd46, 0x3c208c16, 0x6871ca8d, 0x97816a91, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72}}};

  static void mul_fp2_field_by_frob_coeff(g2_point_field_t& fe, unsigned power)
  {
    fe.c1 = fe.c1 * BASE_FIELD_FROBENIUS_COEFF_C1[power % 2];
  }

  struct fq6_config {
    // nonresidue to generate the extension field
    static constexpr g2_point_field_t nonresidue = g2_point_field_t{point_field_t::from(9), point_field_t::one()};
    // true if nonresidue is negative
    static constexpr bool nonresidue_is_negative = false;
    static constexpr bool nonresidue_is_u32 = false;

    static constexpr g2_point_field_t FROBENIUS_COEFF_C1[4] = {
      {point_field_t::one(), point_field_t::zero()},
      {{{0x176f553d, 0x99e39557, 0xc2c3330c, 0xb78cc310, 0xf559b143, 0x4c0bec3c, 0x4f7911f7, 0x2fb34798}},
       {0x640fcba2, 0x1665d51c, 0x0b7c9dce, 0x32ae2a1d, 0xd75a0794, 0x4ba4cc8b, 0x61ebae20, 0x16c9e550}},
      {{{0x607cfd48, 0xe4bd44e5, 0xbb966e3d, 0xc28f069f, 0xe0acccb0, 0x5e6dd9e7, 0xe131a029, 0x30644e72}},
       point_field_t::zero()},
      {{{0x7bdcfb6d, 0x7b746ee8, 0x5d6942d3, 0x805ffd3d, 0x959f25ac, 0xbaff1c77, 0xb755ef0a, 0x0856e078}},
       {{0xaaa586de, 0x380cab2b, 0x98ff2631, 0x0fdf31bf, 0xec26094f, 0xa9f30e6d, 0xb3d1766f, 0x04f1de41}}}};

    static constexpr g2_point_field_t FROBENIUS_COEFF_C2[4] = {
      {point_field_t::one(), point_field_t::zero()},
      {{{0x921ea762, 0x848a1f55, 0xbe94ec72, 0xd33365f7, 0x5a181e84, 0x80f3c0b7, 0x64eea801, 0x05b54f5e}},
       {{0xcd2b8126, 0xc13b4711, 0x1bdec763, 0x3685d2ea, 0x3b0b1c92, 0x9f3a80b0, 0xe7fd8aee, 0x2c145edb}}},
      {{{0x77fffffe, 0x57634731, 0xacdb5c4f, 0xd4f263f1, 0xa0d48bac, 0x59e26bce}}, point_field_t::zero()},
      {{{0x3ccbf066, 0x0e1a92bc, 0x75b06bcb, 0xe6330945, 0xb5b2444e, 0x19bee0f7, 0x11c08dab, 0x0bc58c66}},
       {{0x730c239f, 0x5fe3ed9d, 0x737f96e5, 0xa44a9e08, 0x0cd21d04, 0xfeb0f6ef, 0xe1910a12, 0x23d5e999}}}};

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

    static constexpr g2_point_field_t FROBENIUS_COEFF_C1[4] = {
      {
        point_field_t::one(),
        point_field_t::zero(),
      },
      {{{0xdcc9e470, 0xd60b35da, 0x292f2176, 0x5c521e08, 0x76e68b60, 0xe8b99fdd, 0x2865a7df, 0x1284b71c}},
       {{0x80f362ac, 0xca5cf05f, 0x8eeec7e5, 0x74799277, 0x12150b8e, 0xa6327cfe, 0xb4fae7e6, 0x246996f3}}},
      {
        {{0x607cfd49, 0xe4bd44e5, 0xbb966e3d, 0xc28f069f, 0xe0acccb0, 0x5e6dd9e7, 0xe131a029, 0x30644e72}},
        point_field_t::zero(),
      },
      {{{0x1ed4a67f, 0xe86f7d39, 0xbe55d24a, 0x894cb38d, 0xd0acaa90, 0xefe9608c, 0xcc82e4bb, 0x19dc81cf}},
       {{0xf4c0c101, 0x7694aa2b, 0x97d439ec, 0x7f03a5e3, 0x3576139d, 0x06cbeee3, 0x0be77d73, 0x00abf8b6}}}};
  };
  typedef ComplexExtensionField<fq12_config, fq6_field_t> fq12_field_t; // T3
  typedef fq12_field_t target_field_t;
} // namespace bn254