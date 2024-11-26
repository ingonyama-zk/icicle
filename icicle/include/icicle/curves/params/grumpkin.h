#pragma once

#include "icicle/curves/projective.h"
#include "icicle/fields/snark_fields/grumpkin_base.h"
#include "icicle/fields/snark_fields/grumpkin_scalar.h"

namespace grumpkin {
  struct G1;
  typedef Field<bn254::fp_config> point_field_t;
  typedef Projective<point_field_t, scalar_t, G1> projective_t;
  typedef Affine<point_field_t> affine_t;
  // G1 generator
  struct G1 {
    static constexpr point_field_t gen_x = {0x00000001, 0x00000000, 0x00000000, 0x00000000,
                                            0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr point_field_t gen_y = {0x823f272c, 0x833fc48d, 0xf1181294, 0x2d270d45,
                                            0x6a45d63,  0xcf135e75, 0x00000002, 0x00000000};

    // static constexpr point_field_t weierstrass_b = {0xeffffff0, 0x43e1f593, 0x79b97091, 0x2833e848,
    //                                                 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
    static constexpr point_field_t weierstrass_b = {0x00000011, 0x00000000, 0x00000000, 0x00000000,
                                                    0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr bool is_b_u32 = true;
    static constexpr bool is_b_neg = true;
  }; // G1
} // namespace grumpkin
