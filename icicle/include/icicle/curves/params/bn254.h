#pragma once

#include "icicle/curves/projective.h"
#include "icicle/fields/snark_fields/bn254_base.h"
#include "icicle/fields/snark_fields/bn254_scalar.h"
#include "icicle/fields/complex_extension.h"

namespace bn254 {
  struct G1;
  typedef Field<fq_config> point_field_t;
  typedef Projective<point_field_t, scalar_t, G1> projective_t;
  typedef Affine<point_field_t> affine_t;

  struct G2;
  typedef ComplexExtensionField<fq_config, point_field_t> g2_point_field_t;
  typedef Projective<g2_point_field_t, scalar_t, G2> g2_projective_t;
  typedef Affine<g2_point_field_t> g2_affine_t;

  // G1 and G2 generators
  struct G1 {
    static constexpr point_field_t gen_x = {0x00000001, 0x00000000, 0x00000000, 0x00000000,
                                            0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr point_field_t gen_y = {0x00000002, 0x00000000, 0x00000000, 0x00000000,
                                            0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr point_field_t weierstrass_b = {0x00000003, 0x00000000, 0x00000000, 0x00000000,
                                                    0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr bool is_b_u32 = true;
    static constexpr bool is_b_neg = false;                                                    
  }; // G1

  struct G2 {
    static constexpr point_field_t g2_gen_x_re = {0xd992f6ed, 0x46debd5c, 0xf75edadd, 0x674322d4,
                                                  0x5e5c4479, 0x426a0066, 0x121f1e76, 0x1800deef};
    static constexpr point_field_t g2_gen_x_im = {0xaef312c2, 0x97e485b7, 0x35a9e712, 0xf1aa4933,
                                                  0x31fb5d25, 0x7260bfb7, 0x920d483a, 0x198e9393};
    static constexpr point_field_t g2_gen_y_re = {0x66fa7daa, 0x4ce6cc01, 0x0c43d37b, 0xe3d1e769,
                                                  0x8dcb408f, 0x4aab7180, 0xdb8c6deb, 0x12c85ea5};
    static constexpr point_field_t g2_gen_y_im = {0xd122975b, 0x55acdadc, 0x70b38ef3, 0xbc4b3133,
                                                  0x690c3395, 0xec9e99ad, 0x585ff075, 0x090689d0};
    static constexpr point_field_t weierstrass_b_g2_re = {0x24a138e5, 0x3267e6dc, 0x59dbefa3, 0xb5b4c5e5,
                                                          0x1be06ac3, 0x81be1899, 0xceb8aaae, 0x2b149d40};
    static constexpr point_field_t weierstrass_b_g2_im = {0x85c315d2, 0xe4a2bd06, 0xe52d1852, 0xa74fa084,
                                                          0xeed8fdf4, 0xcd2cafad, 0x3af0fed4, 0x009713b0};

    static constexpr g2_point_field_t gen_x = {g2_gen_x_re, g2_gen_x_im};
    static constexpr g2_point_field_t gen_y = {g2_gen_y_re, g2_gen_y_im};
    static constexpr g2_point_field_t weierstrass_b = {weierstrass_b_g2_re, weierstrass_b_g2_im};
  };

} // namespace bn254
