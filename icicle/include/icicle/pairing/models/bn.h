#pragma once

#include "icicle/pairing/models/bls12.h"

namespace icicle_bn_pairing {
  using icicle_bls12_pairing::add_in_place;
  using icicle_bls12_pairing::double_in_place;
  using icicle_bls12_pairing::ell;
  using icicle_bls12_pairing::exp_by_z;
  using icicle_bls12_pairing::TwistType;

  template <typename Config>
  typename Config::Fp12
  miller_loop(const typename Config::G1Affine& p, const std::vector<typename Config::Fp6>& q_coeffs)
  {
    using Fp12 = typename Config::Fp12;

    Fp12 f = Fp12::one();
    int i = 0;

    constexpr size_t ate_loop_bits = sizeof(Config::ATE_LOOP_BITS) / sizeof(int);
    for (int j = ate_loop_bits - 2; j >= 0; j--) {
      if (j != ate_loop_bits - 1) { f = Fp12::sqr(f); }
      ell<Config>(f, q_coeffs[i++], p);
      if (Config::ATE_LOOP_BITS[j]) { ell<Config>(f, q_coeffs[i++], p); }
    }

    if (Config::Z_IS_NEGATIVE) { f.c1 = -f.c1; }
    ell<Config>(f, q_coeffs[i++], p); // q1
    ell<Config>(f, q_coeffs[i++], p); // q2
    return f;
  }

  template <typename Config>
  void final_exponentiation(typename Config::Fp12& f)
  {
    // https://eprint.iacr.org/2020/875
    using Fp12 = typename Config::Fp12;

    // f1 = f^(p^6)
    Fp12 f1 = f;
    f1.c1 = -f1.c1;

    // f2 = f^(-1)
    Fp12 f2 = Fp12::inverse(f);
    if (f2 == Fp12::zero()) { // Handle potential inverse failure if necessary
      f = Fp12::one();        // Or some other defined behavior for non-invertible input
      return;
    }

    // r = f^(p^6 - 1)
    Fp12 r = f1 * f2;

    // f2 = f^(p^6 - 1)
    f2 = r;
    // r = f^((p^6 - 1)(p^2))
    Config::frobenius_map(r, 2);

    // r = f^((p^6 - 1)(p^2 + 1))
    r *= f2;

    // Hard part from Laura Fuentes-Castaneda et al. "Faster hashing to G2"

    Fp12 y0 = exp_by_z<Config, true>(r);
    Fp12 y1 = Fp12::sqr(y0);
    Fp12 y2 = Fp12::sqr(y1);
    Fp12 y3 = y2 * y1;
    Fp12 y4 = exp_by_z<Config, true>(y3);
    Fp12 y5 = Fp12::sqr(y4);
    Fp12 y6 = exp_by_z<Config, true>(y5);

    y3.c1 = -y3.c1;
    y6.c1 = -y6.c1;

    Fp12 y7 = y6 * y4;
    Fp12 y8 = y7 * y3;
    Fp12 y9 = y8 * y1;
    Fp12 y10 = y8 * y4;
    Fp12 y11 = y10 * r;
    Fp12 y12 = y9;
    Config::frobenius_map(y12, 1);
    Fp12 y13 = y12 * y11;
    Config::frobenius_map(y8, 2);
    Fp12 y14 = y8 * y13;

    r.c1 = -r.c1;

    Fp12 y15 = r * y9;
    Config::frobenius_map(y15, 3);
    Fp12 y16 = y15 * y14;

    f = y16;
  }

  template <typename Config>
  typename Config::G2Affine mul_by_char(typename Config::G2Affine r)
  {
    typename Config::G2Affine s = r;
    Config::mul_fp2_field_by_frob_coeff(s.x, 1);
    s.x *= Config::TWIST_MUL_BY_Q_X;
    Config::mul_fp2_field_by_frob_coeff(s.y, 1);
    s.y *= Config::TWIST_MUL_BY_Q_Y;

    return s;
  }

  template <typename Config>
  std::vector<typename Config::Fp6> prepare_q(const typename Config::G2Affine& q)
  {
    typename Config::Fp two_inv = Config::Fp::inverse(Config::Fp::one() + Config::Fp::one());
    std::vector<typename Config::Fp6> coeffs;
    typename Config::Fp6 r = {q.x, q.y, Config::Fp2::one()};

    typename Config::G2Affine neg_q =
      Config::G2Projective::to_affine(Config::G2Projective::neg(Config::G2Projective::from_affine(q)));

    constexpr size_t ate_loop_bits = sizeof(Config::ATE_LOOP_BITS) / sizeof(int);
    for (int i = ate_loop_bits - 2; i >= 0; i--) {
      coeffs.push_back(double_in_place<Config>(r, two_inv));
      int bit = Config::ATE_LOOP_BITS[i];
      if (bit == 1) {
        coeffs.push_back(add_in_place<Config>(r, q));
      } else if (bit == -1) {
        coeffs.push_back(add_in_place<Config>(r, neg_q));
      }
    }

    typename Config::G2Affine q1 = mul_by_char<Config>(q);
    typename Config::G2Affine q2 = mul_by_char<Config>(q1);

    q2.y = Config::Fp2::neg(q2.y);

    coeffs.push_back(add_in_place<Config>(r, q1));
    coeffs.push_back(add_in_place<Config>(r, q2));

    return coeffs;
  }
} // namespace icicle_bn_pairing