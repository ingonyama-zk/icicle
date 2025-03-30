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

// #ifdef PAIRING_ENABLED

  // TODO: Split this exponent into 4 exponentiations using basis precomputation
  static const storage<40> PAIRING_EXP_LEFT = {
    0x38e3ba79, 0xe516c3f4, 0xe208ccf1, 0xfa9912aa, 0x335d5b68, 0x905ce937, 0xb0dea236, 0xc71a2629, 0x996754c8,
    0x83774940, 0xb6a1e799, 0x21d160ae, 0xed237db4, 0x2ed0b283, 0x6c6f1821, 0x915c97f3, 0xde783765, 0x67f17fcb,
    0x9096d1b7, 0x2378b903, 0x1bdc51dc, 0x7988f876, 0x03fc77a1, 0x20769950, 0xa621315b, 0x827eca0b, 0x8d63cb9f,
    0xe5a72bce, 0xc28b6f8a, 0xf68f7764, 0xcf081517, 0x2f230063, 0x528d6a9a, 0x94506632, 0xeb996ca3, 0xd3cde88e,
    0x195c899e, 0xc0bd38c3, 0x3d807d01, 0x000f686b};

  struct PairingImpl {
    static constexpr scalar_t::ff_storage R = scalar_t::get_modulus();
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
      static constexpr fq6_field_t nonresidue = fq6_field_t{g2_point_field_t::zero(), g2_point_field_t::one(), g2_point_field_t::zero()};
      // true if nonresidue is negative
      static constexpr bool nonresidue_is_negative = false;
      static constexpr bool nonresidue_is_u32 = false;
    };
    typedef ComplexExtensionField<fq12_config, fq6_field_t> fq12_field_t; // T3
    typedef fq12_field_t target_field_t;

    typedef Projective<target_field_t, scalar_t, G1> target_projective_t;
    typedef Affine<target_field_t> target_affine_t;

    static target_affine_t untwist(const g2_affine_t& p1) {
      g2_point_field_t two_inv = g2_point_field_t::inverse(g2_point_field_t::one() + g2_point_field_t::one());
      g2_point_field_t one_minus_i = g2_point_field_t{point_field_t::one(), point_field_t::neg(point_field_t::one())};
      g2_point_field_t coeff = two_inv * one_minus_i; // coeff = (1 - i) / 2
      target_affine_t p2 = target_affine_t::zero(); // p2 = (X, Y)

      p2.x.c0.c2 = p1.x * coeff; // p2.X = (p1.x * coeff * v^2) + 0 * u
      p2.y.c1.c1 = p1.y * coeff; // p2.Y = 0 + (p1.y * coeff * v) * u

      return p2;
    }

    static void final_exponentiation(target_field_t &f) {
      f = target_field_t::pow(f, PAIRING_EXP_LEFT);
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
