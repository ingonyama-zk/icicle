#pragma once

namespace icicle_bls12_pairing {
  enum class TwistType { M, D };

  namespace {
    template <typename Config>
    typename Config::Fp6 double_in_place(typename Config::Fp6& r, const typename Config::Fp& two_inv)
    {
      using Fp2 = typename Config::Fp2;
      using Fp6 = typename Config::Fp6;

      Fp2& x = r.c0;
      Fp2& y = r.c1;
      Fp2& z = r.c2;

      Fp2 a = x * y;
      a *= two_inv;

      Fp2 b = Fp2::sqr(y);
      Fp2 c = Fp2::sqr(z);
      Fp2 e = Config::G2Config::weierstrass_b * (c + c + c);
      Fp2 f = e + e + e;
      Fp2 g = (b + f) * two_inv;
      Fp2 h = Fp2::sqr(y + z) - (b + c);
      Fp2 i = e - b;
      Fp2 j = Fp2::sqr(x);
      Fp2 e_square = Fp2::sqr(e);

      x = a * (b - f);
      y = Fp2::sqr(g) - (e_square + e_square + e_square);
      z = b * h;

      if (Config::TWIST_TYPE == TwistType::M) {
        return Fp6{i, j + j + j, -h};
      } else { // TwistType::D
        return Fp6{-h, j + j + j, i};
      }
    }

    template <typename Config>
    typename Config::Fp6 add_in_place(typename Config::Fp6& r, const typename Config::G2Affine& q)
    {
      using Fp2 = typename Config::Fp2;
      using Fp6 = typename Config::Fp6;

      Fp2& x = r.c0;
      Fp2& y = r.c1;
      Fp2& z = r.c2;

      Fp2 theta = y - (q.y * z);
      Fp2 lambda = x - (q.x * z);
      Fp2 c = Fp2::sqr(theta);
      Fp2 d = Fp2::sqr(lambda);
      Fp2 e = lambda * d;
      Fp2 f = z * c;
      Fp2 g = x * d;
      Fp2 h = e + f - (g + g);

      x = lambda * h;
      y = theta * (g - h) - (e * y);
      z *= e;

      Fp2 j = theta * q.x - (lambda * q.y);
      if (Config::TWIST_TYPE == TwistType::M) {
        return Fp6{j, -theta, lambda};
      } else { // TwistType::D
        return Fp6{lambda, -theta, j};
      }
    }

    template <typename Config>
    void mul_by_1(typename Config::Fp6& r, const typename Config::Fp2& c1)
    {
      using Fp2 = typename Config::Fp2;

      Fp2 b_b = r.c1;
      b_b *= c1;

      Fp2 t1 = c1;
      Fp2 tmp = r.c1;
      tmp += r.c2;
      t1 *= tmp;
      t1 -= b_b;
      t1 *= Config::CUBIC_NONRESIDUE;

      Fp2 t2 = c1;
      tmp = r.c0;
      tmp += r.c1;
      t2 *= tmp;
      t2 -= b_b;

      r.c0 = t1;
      r.c1 = t2;
      r.c2 = b_b;
    }

    template <typename Config>
    void mul_by_01(typename Config::Fp6& r, const typename Config::Fp2& c0, const typename Config::Fp2& c1)
    {
      using Fp2 = typename Config::Fp2;

      Fp2 a_a = r.c0;
      Fp2 b_b = r.c1;
      a_a *= c0;
      b_b *= c1;

      Fp2 t1 = c1;
      Fp2 tmp = r.c1;
      tmp += r.c2;
      t1 *= tmp;
      t1 -= b_b;
      t1 *= Config::CUBIC_NONRESIDUE;
      t1 += a_a;

      Fp2 t3 = c0;
      tmp = r.c0;
      tmp += r.c2;
      t3 *= tmp;
      t3 -= a_a;
      t3 += b_b;

      Fp2 t2 = c0;
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

    template <typename Config>
    void mul_by_014(
      typename Config::Fp12& f,
      const typename Config::Fp2& c0,
      const typename Config::Fp2& c1,
      const typename Config::Fp2& c4)
    {
      using Fp2 = typename Config::Fp2;
      using Fp6 = typename Config::Fp6;

      Fp6 aa = f.c0;
      mul_by_01<Config>(aa, c0, c1);

      Fp6 bb = f.c1;
      mul_by_1<Config>(bb, c4);

      Fp2 o = c1;
      o += c4;

      f.c1 += f.c0;
      mul_by_01<Config>(f.c1, c0, o);
      f.c1 -= aa;
      f.c1 -= bb;

      f.c0 = bb;
      Fp2 old_c1 = f.c0.c1;
      f.c0.c1 = f.c0.c0;
      f.c0.c0 = f.c0.c2;
      f.c0.c0 = f.c0.c0 * Config::CUBIC_NONRESIDUE;
      f.c0.c2 = old_c1;
      f.c0 += aa;
    }

    template <typename Config>
    void mul_by_034(
      typename Config::Fp12& f,
      const typename Config::Fp2& c0,
      const typename Config::Fp2& c3,
      const typename Config::Fp2& c4)
    {
      using Fp2 = typename Config::Fp2;
      using Fp6 = typename Config::Fp6;

      Fp2 a0 = f.c0.c0 * c0;
      Fp2 a1 = f.c0.c1 * c0;
      Fp2 a2 = f.c0.c2 * c0;
      Fp6 a = {a0, a1, a2};

      Fp6 b = f.c1;
      mul_by_01<Config>(b, c3, c4);

      Fp2 c0_plus_c3 = c0 + c3;

      Fp6 e = f.c0 + f.c1;
      mul_by_01<Config>(e, c0_plus_c3, c4);

      f.c1 = e - (a + b);

      f.c0 = b;
      Config::mul_fp6_by_nonresidue(f.c0);
      f.c0 += a;
    }
  } // namespace

  // Evaluate at p
  template <typename Config>
  void ell(typename Config::Fp12& f, typename Config::Fp6 coeffs, typename Config::G1Affine p)
  {
    using Fp2 = typename Config::Fp2;

    Fp2 c0 = coeffs.c0;
    Fp2 c1 = coeffs.c1;
    Fp2 c2 = coeffs.c2;

    if (Config::TWIST_TYPE == TwistType::M) {
      c2 *= p.y;
      c1 *= p.x;
      mul_by_014<Config>(f, c0, c1, c2);
    } else {
      c0 *= p.y;
      c1 *= p.x;
      mul_by_034<Config>(f, c0, c1, c2);
    }
  }

  // cyclotomic exponentiation
  template <typename Config, bool Negate = false>
  typename Config::Fp12 exp_by_z(typename Config::Fp12& f)
  {
    using Fp12 = typename Config::Fp12;

    Fp12 res = Fp12::one();
    Fp12 f_inv = f;
    f_inv.c1 = -f_inv.c1;
    bool nonzero_bit = false;

    constexpr size_t znaf_bits = sizeof(Config::Z_NAF) / sizeof(int);
    for (int i = znaf_bits - 1; i >= 0; i--) {
      if (nonzero_bit) { res = Fp12::sqr(res); }
      int bit = Config::Z_NAF[i];
      if (bit != 0) {
        nonzero_bit = true;
        if (bit > 0) {
          res *= f;
        } else {
          res *= f_inv;
        }
      }
    }

    if (Config::Z_IS_NEGATIVE || (!Config::Z_IS_NEGATIVE && Negate)) { res.c1 = -res.c1; }
    return res;
  }

  template <typename Config>
  typename Config::Fp12
  miller_loop(const typename Config::G1Affine& p, const std::vector<typename Config::Fp6>& q_coeffs)
  {
    using Fp12 = typename Config::Fp12;

    Fp12 f = Fp12::one();
    int i = 0;

    for (int j = sizeof(Config::Z) * 8 - 1; j > 0; j--) {
      f = Fp12::sqr(f);
      ell<Config>(f, q_coeffs[i++], p);
      if (host_math::get_bit(Config::Z, j - 1)) { ell<Config>(f, q_coeffs[i++], p); }
    }

    if (Config::Z_IS_NEGATIVE) { f.c1 = -f.c1; }
    return f;
  }

  template <typename Config>
  void final_exponentiation(typename Config::Fp12& f)
  {
    // https://eprint.iacr.org/2020/875
    using Fp12 = typename Config::Fp12;

    Fp12 f1 = f;
    f1.c1 = -f1.c1;
    Fp12 f2 = Fp12::inverse(f);
    Fp12 r = f1 * f2;
    f2 = r;
    Config::frobenius_map(r, 2);
    r *= f2;
    Fp12 y0 = Fp12::sqr(r);
    Fp12 y1 = exp_by_z<Config>(r);
    Fp12 y2 = r;
    y2.c1 = -y2.c1;
    y1 *= y2;
    y2 = exp_by_z<Config>(y1);
    y1.c1 = -y1.c1;
    y1 *= y2;
    y2 = exp_by_z<Config>(y1);
    Config::frobenius_map(y1, 1);
    y1 *= y2;
    r *= y0;
    y0 = exp_by_z<Config>(y1);
    y2 = exp_by_z<Config>(y0);
    y0 = y1;
    Config::frobenius_map(y0, 2);
    y1.c1 = -y1.c1;
    y1 *= y2;
    y1 *= y0;
    r *= y1;

    f = r;
  }

  template <typename Config>
  std::vector<typename Config::Fp6> prepare_q(const typename Config::G2Affine& q)
  {
    typename Config::Fp two_inv = Config::Fp::inverse(Config::Fp::one() + Config::Fp::one());
    std::vector<typename Config::Fp6> coeffs;
    typename Config::Fp6 r = {q.x, q.y, Config::Fp2::one()};

    for (int j = sizeof(Config::Z) * 8 - 1; j > 0; j--) {
      coeffs.push_back(double_in_place<Config>(r, two_inv));
      if (host_math::get_bit(Config::Z, j - 1)) { coeffs.push_back(add_in_place<Config>(r, q)); }
    }

    return coeffs;
  }
} // namespace icicle_bls12_pairing