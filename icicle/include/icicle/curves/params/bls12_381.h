#pragma once

#include "icicle/curves/projective.h"
#include "icicle/fields/snark_fields/bls12_381_base.h"
#include "icicle/fields/snark_fields/bls12_381_scalar.h"
#include "icicle/fields/complex_extension.h"
#include "icicle/fields/cubic_extension.h"

namespace bls12_381 {
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
    static constexpr point_field_t gen_x = {0xdb22c6bb, 0xfb3af00a, 0xf97a1aef, 0x6c55e83f, 0x171bac58, 0xa14e3a3f,
                                            0x9774b905, 0xc3688c4f, 0x4fa9ac0f, 0x2695638c, 0x3197d794, 0x17f1d3a7};
    static constexpr point_field_t gen_y = {0x46c5e7e1, 0x0caa2329, 0xa2888ae4, 0xd03cc744, 0x2c04b3ed, 0x00db18cb,
                                            0xd5d00af6, 0xfcf5e095, 0x741d8ae4, 0xa09e30ed, 0xe3aaa0f1, 0x08b3f481};
    static constexpr point_field_t weierstrass_b = {0x00000004, 0x00000000, 0x00000000, 0x00000000,
                                                    0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                                    0x00000000, 0x00000000, 0x00000000, 0x00000000};

    static constexpr bool is_b_u32 = true;
    static constexpr bool is_b_neg = false;
  };

  struct G2 {
    static constexpr point_field_t g2_gen_x_re = {0xc121bdb8, 0xd48056c8, 0xa805bbef, 0x0bac0326,
                                                  0x7ae3d177, 0xb4510b64, 0xfa403b02, 0xc6e47ad4,
                                                  0x2dc51051, 0x26080527, 0xf08f0a91, 0x024aa2b2};
    static constexpr point_field_t g2_gen_x_im = {0x5d042b7e, 0xe5ac7d05, 0x13945d57, 0x334cf112,
                                                  0xdc7f5049, 0xb5da61bb, 0x9920b61a, 0x596bd0d0,
                                                  0x88274f65, 0x7dacd3a0, 0x52719f60, 0x13e02b60};
    static constexpr point_field_t g2_gen_y_re = {0x08b82801, 0xe1935486, 0x3baca289, 0x923ac9cc,
                                                  0x5160d12c, 0x6d429a69, 0x8cbdd3a7, 0xadfd9baa,
                                                  0xda2e351a, 0x8cc9cdc6, 0x727d6e11, 0x0ce5d527};
    static constexpr point_field_t g2_gen_y_im = {0xf05f79be, 0xaaa9075f, 0x5cec1da1, 0x3f370d27,
                                                  0x572e99ab, 0x267492ab, 0x85a763af, 0xcb3e287e,
                                                  0x2bc28b99, 0x32acd2b0, 0x2ea734cc, 0x0606c4a0};

    static constexpr point_field_t weierstrass_b_g2_re = {0x00000004, 0x00000000, 0x00000000, 0x00000000,
                                                          0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                                          0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr point_field_t weierstrass_b_g2_im = {0x00000004, 0x00000000, 0x00000000, 0x00000000,
                                                          0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                                          0x00000000, 0x00000000, 0x00000000, 0x00000000};

    static constexpr bool is_b_u32_g2_re = true;
    static constexpr bool is_b_neg_g2_re = false;
    static constexpr bool is_b_u32_g2_im = true;
    static constexpr bool is_b_neg_g2_im = false;

    static constexpr g2_point_field_t gen_x = {g2_gen_x_re, g2_gen_x_im};
    static constexpr g2_point_field_t gen_y = {g2_gen_y_re, g2_gen_y_im};
    static constexpr g2_point_field_t weierstrass_b = {weierstrass_b_g2_re, weierstrass_b_g2_im};
  };

  // TODO: Uncomment
  // #ifdef PAIRING_ENABLED

  // TODO: Remove
  static const storage<136> PAIRING_EXP_FULL = {
    0x5df57510, 0xc0bcb9b5, 0xe68bfb24, 0x25f98630, 0xfbd5f489, 0x4406fbc8, 0xd12191a0, 0x8e2f8491, 0x0a6f8069, 0x3e9d7165, 0x1d4cab80, 0x226c2f01, 0x17489119, 0x67f67c47, 0xd88592d7, 0xaf3f881b, 0xeed2161d, 0x1a67e49e, 0x69aeb218, 0xe5b78c78, 0x043f7bbc, 0xf6539314, 0xf2701aae, 0x73f62537, 0xe9622d2a, 0xaff1c910, 0x92caa9d4, 0x62833134, 0xbea83d19, 0x2e2f3ec2, 0xb02faa73, 0xa4c7e79f, 0xd7961be1, 0x6c49637f, 0xe8817745, 0x08e88adc, 0x36399917, 0x35de3f7a, 0x31759c36, 0x9c1d9f7c, 0x4ea820b0, 0xfa9e13c2, 0xa403577d, 0x3fc56947, 0xfc5cceb7, 0xa4c1b6dc, 0x7066bca6, 0x1bbd8136, 0x0bc62775, 0x0418a3ef, 0xa9f9e010, 0x49bf9b71, 0x7db60b17, 0x51129109, 0xe5308f1c, 0x498345c6, 0x9dadd7c2, 0x6d8823b1, 0xd556952c, 0x92004ced, 0xc03ef195, 0x4c6bec3e, 0x044ce6ad, 0x0a1fad20, 0xcd15948d, 0xc55d3109, 0x2c3f0bd0, 0x334f46c0, 0x34c05739, 0x3b5a62eb, 0x1d1676a5, 0x72453841, 0xd0463434, 0x127a1b5a, 0xc85b0129, 0x61a474c5, 0x86ef965e, 0x8dfc8e28, 0x459f1243, 0x96532fef, 0xcdc10412, 0x40ee7169, 0xb74bb22a, 0x9c40a68e, 0xf4684d0b, 0x25118790, 0xc8d4c01f, 0x596bc293, 0x27611212, 0x1064837f, 0xbf24dde4, 0x077ffb10, 0xcd2b01f3, 0xc49f570b, 0x4c374693, 0x1a0c5bf2, 0x9bc73ab6, 0x350da535, 0xe4d7acdd, 0xd2670d93, 0x6e1ab656, 0xd39099b8, 0x978e2b0d, 0x19328148, 0x386b0e88, 0xb113f414, 0x630d9aa4, 0x07a0dce2, 0x93753318, 0xa927e7bb, 0xad49466f, 0xe347aa68, 0x106feaf4, 0x1c0ad0d6, 0xff3a0f0f, 0xc872ee83, 0xa660835c, 0x074e43b9, 0xe9cfee9a, 0xc0aadff5, 0xc7deada9, 0x30698e8c, 0xab353f2c, 0xd1073776, 0xbadc3a43, 0x17848517, 0x3f8d14a9, 0x7363baa1, 0x7d4507d0, 0xd4977b3f, 0x89ee0193, 0x496a1c0a, 0xe1bda9c0, 0xdcc825b7, 0x02ee1db5, 0x00000000
  };

  // TODO: Split this exponent into 4 exponentiations using basis precomputation
  static const storage<40> PAIRING_EXP_LEFT = {
    0x38e3ba79, 0xe516c3f4, 0xe208ccf1, 0xfa9912aa, 0x335d5b68, 0x905ce937, 0xb0dea236, 0xc71a2629,
    0x996754c8, 0x83774940, 0xb6a1e799, 0x21d160ae, 0xed237db4, 0x2ed0b283, 0x6c6f1821, 0x915c97f3,
    0xde783765, 0x67f17fcb, 0x9096d1b7, 0x2378b903, 0x1bdc51dc, 0x7988f876, 0x03fc77a1, 0x20769950,
    0xa621315b, 0x827eca0b, 0x8d63cb9f, 0xe5a72bce, 0xc28b6f8a, 0xf68f7764, 0xcf081517, 0x2f230063,
    0x528d6a9a, 0x94506632, 0xeb996ca3, 0xd3cde88e, 0x195c899e, 0xc0bd38c3, 0x3d807d01, 0x000f686b};

  struct PairingImpl {
    static constexpr scalar_t::ff_storage R = scalar_t::get_modulus();
    static constexpr scalar_t::ff_storage RPRIME = {0x00000000, 0xffffffff, 0xfffe5bfe, 0x53bda402,
                                                    0x09a1d805, 0x3339d808, 0x299d7d48, 0x73eda753};
    static constexpr unsigned R_BITS = scalar_t::NBITS;
    static constexpr unsigned R_LIMBS = scalar_t::TLC;

    struct fq6_config {
      // nonresidue to generate the extension field
      static constexpr g2_point_field_t nonresidue = g2_point_field_t{point_field_t::one(), point_field_t::one()};
      // true if nonresidue is negative
      static constexpr bool nonresidue_is_negative = false;
      static constexpr bool nonresidue_is_u32 = false;
    };
    typedef CubicExtensionField<fq6_config, g2_point_field_t> fq6_field_t; // T2

    struct fq12_config {
      // nonresidue to generate the extension field
      static constexpr fq6_field_t nonresidue =
        fq6_field_t{g2_point_field_t::zero(), g2_point_field_t::one(), g2_point_field_t::zero()};
      // true if nonresidue is negative
      static constexpr bool nonresidue_is_negative = false;
      static constexpr bool nonresidue_is_u32 = false;
    };
    typedef ComplexExtensionField<fq12_config, fq6_field_t> fq12_field_t; // T3
    typedef fq12_field_t target_field_t;

    typedef Projective<target_field_t, scalar_t, G1> target_projective_t;
    typedef Affine<target_field_t> target_affine_t;

    static target_affine_t untwist(const g2_affine_t& p1)
    {
      g2_point_field_t two_inv = g2_point_field_t::inverse(g2_point_field_t::one() + g2_point_field_t::one());
      g2_point_field_t one_minus_i = g2_point_field_t{point_field_t::one(), point_field_t::neg(point_field_t::one())};
      g2_point_field_t coeff = two_inv * one_minus_i; // coeff = (1 - i) / 2
      target_affine_t p2 = target_affine_t::zero();   // p2 = (X, Y)

      p2.x.c0.c2 = p1.x * coeff; // p2.X = (p1.x * coeff * v^2) + 0 * u
      p2.y.c1.c1 = p1.y * coeff; // p2.Y = 0 + (p1.y * coeff * v) * u

      return p2;
    }

    static void final_exponentiation(target_field_t& f)
    {
      point_field_t::ff_storage q = point_field_t::get_modulus();

      // f ^ (q^2)
      target_field_t fq2 = f;
      fq2 = target_field_t::pow(fq2, q);
      fq2 = target_field_t::pow(fq2, q);

      // f ^ (q^6)
      target_field_t fq6 = fq2;
      for (int i = 0; i < 4; i++) {
        fq6 = target_field_t::pow(fq6, q);
      }

      // f ^ (q^8)
      target_field_t fq8 = fq6;
      fq8 = target_field_t::pow(fq8, q);
      fq8 = target_field_t::pow(fq8, q);

      f = fq8 * fq6 * target_field_t::inverse(fq2 * f); // f ^ (q^8 + q^6 - (q^2 + 1))
    }
  };
  // #endif
} // namespace bls12_381
