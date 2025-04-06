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

  struct PairingImpl {
    static constexpr scalar_t::ff_storage R = scalar_t::get_modulus();
    static constexpr unsigned R_BITS = scalar_t::NBITS;

    static constexpr storage<2> Z = {0x00010000, 0xd2010000};
    static constexpr unsigned Z_BITS = 64;
    static constexpr bool Z_IS_NEGATIVE = true;
    static constexpr int Z_NAF[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0,
                                    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, -1, 0, 1};
    static constexpr unsigned Z_NAF_BITS = 65;

    static constexpr point_field_t BASE_FIELD_FROBENIUS_COEFF_C1[2] = {
      point_field_t::one(),
      {{0xffffaaaa, 0xb9feffff, 0xb153ffff, 0x1eabfffe, 0xf6b0f624, 0x6730d2a0, 0xf38512bf, 0x64774b84, 0x434bacd7,
        0x4b1ba7b6, 0x397fe69a, 0x1a0111ea}}};

    static HOST_DEVICE void mul_fp2_field_by_frob_coeff(g2_point_field_t& fe, unsigned power)
    {
      // std::cout << "COEFF: " << BASE_FIELD_FROBENIUS_COEFF_C1[power % 2] << std::endl;;
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

      static HOST_DEVICE void
      frobenius_map(g2_point_field_t& c0, g2_point_field_t& c1, g2_point_field_t& c2, unsigned power)
      {
        mul_fp2_field_by_frob_coeff(c0, power);
        mul_fp2_field_by_frob_coeff(c1, power);
        mul_fp2_field_by_frob_coeff(c2, power);
        // std::cout << "CUBIC c1: " << c1 << std::endl;
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

      static HOST_DEVICE void frobenius_map(fq6_field_t& c0, fq6_field_t& c1, unsigned power)
      {
        fq6_config::frobenius_map(c0.c0, c0.c1, c0.c2, power);
        fq6_config::frobenius_map(c1.c0, c1.c1, c1.c2, power);
        c1 *= FROBENIUS_COEFF_C1[power % 12];
      }
    };
    typedef ComplexExtensionField<fq12_config, fq6_field_t> fq12_field_t; // T3
    typedef fq12_field_t target_field_t;

    typedef Projective<target_field_t, scalar_t, G1> target_projective_t;
    typedef Affine<target_field_t> target_affine_t;

  private:
    // cyclotomic exponentiation
    static target_field_t exp_by_z(target_field_t& f)
    {
      target_field_t res = target_field_t::one();
      target_field_t f_inv = f;
      f_inv.c1 = -f_inv.c1;
      bool nonzero_bit = false;
      for (int i = Z_NAF_BITS - 1; i >= 0; i--) {
        if (nonzero_bit) { res = target_field_t::sqr(res); }
        int bit = Z_NAF[i];
        if (bit != 0) {
          nonzero_bit = true;
          if (bit > 0) {
            res *= f;
          } else {
            res *= f_inv;
          }
        }
      }

      res.c1 = -res.c1;
      return res;
    }

    static fq6_field_t double_in_place(fq6_field_t& r, const point_field_t& two_inv)
    {
      g2_point_field_t& x = r.c0;
      g2_point_field_t& y = r.c1;
      g2_point_field_t& z = r.c2;

      g2_point_field_t a = x * y;
      a *= two_inv;

      g2_point_field_t b = g2_point_field_t::sqr(y);
      g2_point_field_t c = g2_point_field_t::sqr(z);
      g2_point_field_t e = G2::weierstrass_b * (c + c + c);
      g2_point_field_t f = e + e + e;
      g2_point_field_t g = (b + f) * two_inv;
      g2_point_field_t h = g2_point_field_t::sqr(y + z) - (b + c);
      g2_point_field_t i = e - b;
      g2_point_field_t j = g2_point_field_t::sqr(x);
      g2_point_field_t e_square = g2_point_field_t::sqr(e);

      x = a * (b - f);
      y = g2_point_field_t::sqr(g) - (e_square + e_square + e_square);
      z = b * h;

      return fq6_field_t{i, j + j + j, -h};
    }

    static fq6_field_t add_in_place(fq6_field_t& r, const g2_affine_t& q)
    {
      g2_point_field_t& x = r.c0;
      g2_point_field_t& y = r.c1;
      g2_point_field_t& z = r.c2;

      g2_point_field_t theta = y - (q.y * z);
      g2_point_field_t lambda = x - (q.x * z);
      g2_point_field_t c = g2_point_field_t::sqr(theta);
      g2_point_field_t d = g2_point_field_t::sqr(lambda);
      g2_point_field_t e = lambda * d;
      g2_point_field_t f = z * c;
      g2_point_field_t g = x * d;
      g2_point_field_t h = e + f - (g + g);

      x = lambda * h;
      y = theta * (g - h) - (e * y);
      z *= e;

      g2_point_field_t j = theta * q.x - (lambda * q.y);
      return fq6_field_t{j, -theta, lambda};
    }

    static void mul_by_1(fq6_field_t& r, const g2_point_field_t& c1)
    {
      g2_point_field_t b_b = r.c1;
      b_b *= c1;

      g2_point_field_t t1 = c1;
      g2_point_field_t tmp = r.c1;
      tmp += r.c2;
      t1 *= tmp;
      t1 -= b_b;
      t1 *= fq6_config::nonresidue;

      g2_point_field_t t2 = c1;
      tmp = r.c0;
      tmp += r.c1;
      t2 *= tmp;
      t2 -= b_b;

      r.c0 = t1;
      r.c1 = t2;
      r.c2 = b_b;
    }

    static void mul_by_01(fq6_field_t& r, const g2_point_field_t& c0, const g2_point_field_t& c1)
    {
      g2_point_field_t a_a = r.c0;
      g2_point_field_t b_b = r.c1;
      a_a *= c0;
      b_b *= c1;

      g2_point_field_t t1 = c1;
      g2_point_field_t tmp = r.c1;
      tmp += r.c2;
      t1 *= tmp;
      t1 -= b_b;
      t1 *= fq6_config::nonresidue;
      t1 += a_a;

      g2_point_field_t t3 = c0;
      tmp = r.c0;
      tmp += r.c2;
      t3 *= tmp;
      t3 -= a_a;
      t3 += b_b;

      g2_point_field_t t2 = c0;
      t2 += c1;
      tmp = r.c0;
      tmp += r.c1;
      t2 *= tmp;
      t2 -= a_a;
      t2 -= b_b;

      r.c0 = t1;
      r.c1 = t2;
      r.c2 = t3;
    }

    static void
    mul_by_014(target_field_t& f, const g2_point_field_t& c0, const g2_point_field_t& c1, const g2_point_field_t& c4)
    {
      fq6_field_t aa = f.c0;
      mul_by_01(aa, c0, c1);

      fq6_field_t bb = f.c1;
      mul_by_1(bb, c4);

      g2_point_field_t o = c1;
      o += c4;

      f.c1 += f.c0;
      mul_by_01(f.c1, c0, o);
      f.c1 -= aa;
      f.c1 -= bb;

      f.c0 = bb;
      g2_point_field_t old_c1 = f.c0.c1;
      f.c0.c1 = f.c0.c0;
      f.c0.c0 = f.c0.c2;
      f.c0.c0 = f.c0.c0 * fq6_config::nonresidue;
      f.c0.c2 = old_c1;
      f.c0 += aa;
    }

    static void ell(target_field_t& f, fq6_field_t coeffs, affine_t p)
    {
      g2_point_field_t c0 = coeffs.c0;
      g2_point_field_t c1 = coeffs.c1;
      g2_point_field_t c2 = coeffs.c2;

      c2 *= p.y;
      c1 *= p.x;
      mul_by_014(f, c0, c1, c2);
    }

  public:
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

    static std::vector<fq6_field_t> prepare_q(const g2_affine_t& q)
    {
      point_field_t two_inv = point_field_t::inverse(point_field_t::one() + point_field_t::one());
      std::vector<fq6_field_t> coeffs;
      fq6_field_t r = fq6_field_t{q.x, q.y, g2_point_field_t::one()};

      for (int j = Z_BITS - 1; j > 0; j--) {
        coeffs.push_back(double_in_place(r, two_inv));
        if (host_math::get_bit(Z, j - 1)) { coeffs.push_back(add_in_place(r, q)); }
      }

      return coeffs;
    }

    static target_field_t opt_miller_loop(const affine_t& p, const std::vector<fq6_field_t> q_coeffs)
    {
      target_field_t f = target_field_t::one();
      int i = 0;
      for (int j = Z_BITS - 1; j > 0; j--) {
        f = target_field_t::sqr(f);
        ell(f, q_coeffs[i++], p);
        if (host_math::get_bit(Z, j - 1)) { ell(f, q_coeffs[i++], p); }
      }

      if (Z_IS_NEGATIVE) { f.c1 = -f.c1; }
      return f;
    }

    static void final_exponentiation(target_field_t& f)
    {
      target_field_t f1 = f;
      f1.c1 = -f1.c1;
      target_field_t f2 = target_field_t::inverse(f);
      target_field_t r = f1 * f2;
      f2 = r;
      fq12_config::frobenius_map(r.c0, r.c1, 2);
      r *= f2;
      target_field_t y0 = target_field_t::sqr(r);
      target_field_t y1 = exp_by_z(r);
      target_field_t y2 = r;
      y2.c1 = -y2.c1;
      y1 *= y2;
      y2 = exp_by_z(y1);
      y1.c1 = -y1.c1;
      y1 *= y2;
      y2 = exp_by_z(y1);
      fq12_config::frobenius_map(y1.c0, y1.c1, 1);
      y1 *= y2;
      r *= y0;
      y0 = exp_by_z(y1);
      y2 = exp_by_z(y0);
      y0 = y1;
      fq12_config::frobenius_map(y0.c0, y0.c1, 2);
      y1.c1 = -y1.c1;
      y1 *= y2;
      y1 *= y0;
      r *= y1;

      f = r;
    }
  };
  // #endif
} // namespace bls12_381
