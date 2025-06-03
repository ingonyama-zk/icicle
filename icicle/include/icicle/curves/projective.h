#pragma once

#ifdef __CUDACC__
  #include "gpu-utils/sharedmem.h"
#endif // __CUDACC__

#include "icicle/utils/modifiers.h"
#include "icicle/curves/affine.h"
#include <array>

#include "icicle/fields/field.h"
// #include "icicle/fields/id.h"
#include "bn254_mont.h"
#include <atomic>

extern std::atomic<uint64_t> prj_cnt;
extern std::atomic<uint64_t> dbl_cnt;
extern std::atomic<uint64_t> mix_cnt;
extern std::atomic<uint64_t> prj_time;
extern std::atomic<uint64_t> dbl_time;
extern std::atomic<uint64_t> mix_time;

inline uint64_t get_clock_counter() {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r" (val));
    return val;
}

inline uint32_t get_counter_frequency() {
    uint32_t freq;
    asm volatile("mrs %0, cntfrq_el0" : "=r" (freq));
    return freq;
}

inline uint64_t counter_to_ns(uint64_t ticks) {
    uint32_t freq = get_counter_frequency();
    return (ticks * 1000000000ULL) / freq;
}

namespace bn254 {
  struct G1;
  struct G2;
}
template <typename FF, class SCALAR_FF, typename Gen>
class Projective
{
  friend Affine<FF>;

public:
  typedef Affine<FF> Aff;
  typedef SCALAR_FF Scalar;

  static constexpr unsigned SCALAR_FF_NBITS = SCALAR_FF::NBITS;
  static constexpr unsigned FF_NBITS = FF::NBITS;

  FF x;
  FF y;
  FF z;
  static HOST_DEVICE_INLINE Projective zero() { return {FF::zero(), FF::one(), FF::zero()}; }

  static HOST_DEVICE_INLINE Affine<FF> to_affine(const Projective& point)
  {
    FF denom = FF::inverse(point.z);
    return {point.x * denom, point.y * denom};
  }
  HOST_DEVICE_INLINE Affine<FF> to_affine() { return to_affine(*this); }

  static HOST_DEVICE_INLINE Projective from_affine(const Affine<FF>& point)
  {
    return point == Affine<FF>::zero() ? zero() : Projective{point.x, point.y, FF::one()};
  }

  static HOST_DEVICE_INLINE Projective to_montgomery(const Projective& point)
  {
    return {FF::to_montgomery(point.x), FF::to_montgomery(point.y), FF::to_montgomery(point.z)};
  }

  static HOST_DEVICE_INLINE Projective from_montgomery(const Projective& point)
  {
    return {FF::from_montgomery(point.x), FF::from_montgomery(point.y), FF::from_montgomery(point.z)};
  }

  static HOST_DEVICE_INLINE Projective generator() { return {Gen::gen_x, Gen::gen_y, FF::one()}; }

  static HOST_DEVICE_INLINE Projective neg(const Projective& point) { return {point.x, FF::neg(point.y), point.z}; }

  // -- [CUSTOM BN254 MONT] ----------------------------------------------------------------------------

  struct c_bn254_t {
    mp_256_t re;
    mp_256_t im;
  };

  struct c_bn254_wide_t {
    mp_512_t re;
    mp_512_t im;
  };

  static __attribute__((always_inline)) mp_256_t mul_b3(const mp_256_t &a) {
    mp_256_t a2 = bn254_mont_t::add_red(a, a);
    mp_256_t a4 = bn254_mont_t::add_red(a2, a2);
    mp_256_t a8 = bn254_mont_t::add_red(a4, a4);
    mp_256_t a9 = bn254_mont_t::add_red(a8, a);
    return a9;
  }

  static c_bn254_t mul_b3(const c_bn254_t &a) {
    constexpr c_bn254_t B3 = {
            {0x1ef69c66bce9b021, 0xf21b7c8d3cb039cf, 0x1499be5e509e8f8f, 0x20753adca9c6bfb8},
            {0xade8371391494176, 0xf5eee18eaf8748f8, 0x67860f09cc8af9dd, 0x1c53b10b0d2fc7e}};
    FF res = reinterpret_cast<const FF &>(a) * reinterpret_cast<const FF &>(B3);
    return reinterpret_cast<c_bn254_t &>(res);
  }

  static c_bn254_t mont_mul(const c_bn254_t &a, const c_bn254_t &b) {
      mp_256_t a_sum = bn254_mont_t::add(a.re, a.im);
      mp_256_t b_sum = bn254_mont_t::add(b.re, b.im);
      mp_256_t re_prod = bn254_mont_t::mul(a.re, b.re);
      mp_256_t im_prod = bn254_mont_t::mul(a.im, b.im);
      mp_256_t sum_prod = bn254_mont_t::mul(a_sum, b_sum);
      return {
              bn254_mont_t::sub_red(re_prod, im_prod),
              bn254_mont_t::sub_red(bn254_mont_t::sub_red(sum_prod, re_prod), im_prod),
      };
  }


  static c_bn254_t mont_reduce(const c_bn254_t &a) {
      return {bn254_mont_t::reduce(a.re), bn254_mont_t::reduce(a.im)};
  }

  static c_bn254_t add(const c_bn254_t &a, const c_bn254_t &b) {
      return {bn254_mont_t::add_red(a.re, b.re), bn254_mont_t::add_red(a.im, b.im)};
  }

  static c_bn254_t add_one(const c_bn254_t &a) {
      return {bn254_mont_t::add_red(a.re, {1, 0, 0, 0}), a.im};
  }

  static c_bn254_t sub(const c_bn254_t &a, const c_bn254_t &b) {
      return {bn254_mont_t::sub_red(a.re, b.re), bn254_mont_t::sub_red(a.im, b.im)};
  }

  static c_bn254_wide_t add(const c_bn254_wide_t &a, const c_bn254_wide_t &b) {
      return {bn254_mont_t::add_red(a.re, b.re), bn254_mont_t::add_red(a.im, b.im)};
  }

  static c_bn254_wide_t add_no_red(const c_bn254_wide_t &a, const c_bn254_wide_t &b) {
      return {bn254_mont_t::add(a.re, b.re), bn254_mont_t::add(a.im, b.im)};
  }

  static c_bn254_wide_t sub(const c_bn254_wide_t &a, const c_bn254_wide_t &b) {
      return {bn254_mont_t::sub_red(a.re, b.re), bn254_mont_t::sub_red(a.im, b.im)};
  }

  static c_bn254_wide_t mul_wide(const c_bn254_t &a, const c_bn254_t &b) {
      mp_256_t a_sum = bn254_mont_t::add(a.re, a.im);
      mp_256_t b_sum = bn254_mont_t::add(b.re, b.im);
      auto re_prod = bn254_mont_t::schoolbook_mul(a.re, b.re);
      auto im_prod = bn254_mont_t::schoolbook_mul(a.im, b.im);
      auto sum_prod = bn254_mont_t::schoolbook_mul(a_sum, b_sum);
      return {
              bn254_mont_t::sub_red(re_prod, im_prod),
              bn254_mont_t::sub_red(bn254_mont_t::sub_red(sum_prod, re_prod), im_prod),
      };
  }

  static c_bn254_t reduce(const c_bn254_wide_t &a) {
      return {bn254_mont_t::reduce(a.re), bn254_mont_t::reduce(a.im)};
  }

  static HOST_DEVICE void accum_prj_aff(Projective &p1, const Affine<FF>& p2) {
    if constexpr (std::is_same_v<Gen, bn254::G1>) {
      const mp_256_t &X1 = reinterpret_cast<mp_256_t &>(p1.x);
      const mp_256_t &Y1 = reinterpret_cast<mp_256_t &>(p1.y);
      const mp_256_t &Z1 = reinterpret_cast<mp_256_t &>(p1.z);
      const mp_256_t &X2 = reinterpret_cast<const mp_256_t &>(p2.x);
      const mp_256_t &Y2 = reinterpret_cast<const mp_256_t &>(p2.y);
      const mp_256_t t00 = bn254_mont_t::mul(X1, X2);
      const mp_256_t t01 = bn254_mont_t::mul(Y1, Y2);
      const mp_256_t t02 = bn254_mont_t::reduce(Z1);
      const mp_256_t t03 = bn254_mont_t::add(X1, Y1);
      const mp_256_t t04 = bn254_mont_t::add(X2, Y2);
      const mp_256_t t05 = bn254_mont_t::mul(t03, t04);
      const mp_256_t t06 = bn254_mont_t::add_red(t00, t01);
      const mp_256_t t07 = bn254_mont_t::sub_red(t05, t06);
      const mp_256_t t08 = bn254_mont_t::add(Y1, Z1);
      const mp_256_t t09 = bn254_mont_t::add(Y2, {1, 0, 0, 0});
      const mp_256_t t10 = bn254_mont_t::mul(t08, t09);
      const mp_256_t t11 = bn254_mont_t::add_red(t01, t02);
      const mp_256_t t12 = bn254_mont_t::sub_red(t10, t11);
      const mp_256_t t13 = bn254_mont_t::add(X1, Z1);
      const mp_256_t t14 = bn254_mont_t::add(X2, {1, 0, 0, 0});
      const mp_256_t t15 = bn254_mont_t::mul(t13, t14);
      const mp_256_t t16 = bn254_mont_t::add_red(t00, t02);
      const mp_256_t t17 = bn254_mont_t::sub_red(t15, t16);
      const mp_256_t t18 = bn254_mont_t::add_red(t00, t00);
      const mp_256_t t19 = bn254_mont_t::add_red(t18, t00);
      const mp_256_t t20 = mul_b3(t02);
      const mp_256_t t21 = bn254_mont_t::add_red(t01, t20);
      const mp_256_t t22 = bn254_mont_t::sub_red(t01, t20);
      const mp_256_t t23 = mul_b3(t17);
      const auto t24 = bn254_mont_t::schoolbook_mul(t12, t23);
      const auto t25 = bn254_mont_t::schoolbook_mul(t07, t22);
      const mp_256_t X3 = bn254_mont_t::reduce(bn254_mont_t::sub_red(t25, t24));
      const auto t27 = bn254_mont_t::schoolbook_mul(t23, t19);
      const auto t28 = bn254_mont_t::schoolbook_mul(t22, t21);
      const mp_256_t Y3 = bn254_mont_t::reduce(bn254_mont_t::add(t28, t27));
      const auto t30 = bn254_mont_t::schoolbook_mul(t19, t07);
      const auto t31 = bn254_mont_t::schoolbook_mul(t21, t12);
      const mp_256_t Z3 = bn254_mont_t::reduce(bn254_mont_t::add(t31, t30));
      p1.x = reinterpret_cast<const FF &>(X3);
      p1.y = reinterpret_cast<const FF &>(Y3);
      p1.z = reinterpret_cast<const FF &>(Z3);
    } else if constexpr (std::is_same_v<Gen, bn254::G2>) {
      const c_bn254_t X1 = reinterpret_cast<const c_bn254_t &>(p1.x);    //                   < 2
      const c_bn254_t Y1 = reinterpret_cast<const c_bn254_t &>(p1.y);    //                   < 2
      const c_bn254_t Z1 = reinterpret_cast<const c_bn254_t &>(p1.z);    //                   < 2
      const c_bn254_t X2 = reinterpret_cast<const c_bn254_t &>(p2.x);    //                   < 2
      const c_bn254_t Y2 = reinterpret_cast<const c_bn254_t &>(p2.y);    //                   < 2
      const c_bn254_t t00 = mont_mul(X1, X2);                     // t00 ← X1 · X2     < 2
      const c_bn254_t t01 = mont_mul(Y1, Y2);                     // t01 ← Y1 · Y2     < 2
      const c_bn254_t t02 = mont_reduce(Z1);                      // t02 ← Z1         < 2
      const c_bn254_t t03 = add(X1, Y1);                          // t03 ← X1 + Y1     < 4
      const c_bn254_t t04 = add(X2, Y2);                          // t04 ← X2 + Y2     < 4
      const c_bn254_t t05 = mont_mul(t03, t04);                   // t03 ← t03 · t04   < 3
      const c_bn254_t t06 = add(t00, t01);                        // t06 ← t00 + t01   < 4
      const c_bn254_t t07 = sub(t05, t06);                        // t05 ← t05 − t06   < 2
      const c_bn254_t t08 = add(Y1, Z1);                          // t08 ← Y1 + Z1     < 4
      const c_bn254_t t09 = add_one(Y2);// t09 ← Y2 + 1      < 4
      const c_bn254_t t10 = mont_mul(t08, t09);                   // t10 ← t08 · t09   < 3
      const c_bn254_t t11 = add(t01, t02);                        // t11 ← t01 + t02   < 4
      const c_bn254_t t12 = sub(t10, t11);                            // t12 ← t10 − t11   < 2
      const c_bn254_t t13 = add(X1, Z1);                              // t13 ← X1 + Z1     < 4
      const c_bn254_t t14 = add_one(X2);                      // t14 ← X2 + 1      < 4
      const c_bn254_t t15 = mont_mul(t13, t14);                   // t15 ← t13 · t14   < 3
      const c_bn254_t t16 = add(t00, t02);                            // t16 ← t00 + t02   < 4
      const c_bn254_t t17 = sub(t15, t16);                            // t17 ← t15 − t16   < 2
      const c_bn254_t t18 = add(t00, t00);                            // t18 ← t00 + t00   < 2
      const c_bn254_t t19 = add(t18, t00);                            // t19 ← t18 + t00   < 2
      const c_bn254_t t20 = mul_b3(t02);                          // t20 ← b3 · t02    < 2
      const c_bn254_t t21 = add(t01, t20);                            // t21 ← t01 + t20   < 2
      const c_bn254_t t22 = sub(t01, t20);                            // t22 ← t01 − t20   < 2
      const c_bn254_t t23 = mul_b3(t17);                          // t23 ← b3 · t17    < 2
      const auto t24 = mul_wide(t12, t23);                        // t24 ← t12 · t23   < 2
      const auto t25 = mul_wide(t07, t22);                        // t25 ← t07 · t22   < 2
      const auto X3 = reduce(sub(t25, t24));                                   // X3 ← t25 − t24    < 2
      const auto t27 = mul_wide(t23, t19);                        // t27 ← t23 · t19   < 2
      const auto t28 = mul_wide(t22, t21);                        // t28 ← t22 · t21   < 2
      const auto Y3 = reduce(add_no_red(t28, t27));                                   // Y3 ← t28 + t27    < 2
      const auto t30 = mul_wide(t19, t07);                        // t30 ← t19 · t07   < 2
      const auto t31 = mul_wide(t21, t12);                        // t31 ← t21 · t12   < 2
      const auto Z3 = reduce(add_no_red(t31, t30));                                   // Z3 ← t31 + t30    < 2
      p1.x = reinterpret_cast<const FF &>(X3);
      p1.y = reinterpret_cast<const FF &>(Y3);
      p1.z = reinterpret_cast<const FF &>(Z3);
    } else {
      const FF X1 = p1.x;                                            //                   < 2
      const FF Y1 = p1.y;                                            //                   < 2
      const FF Z1 = p1.z;                                            //                   < 2
      const FF X2 = p2.x;                                            //                   < 2
      const FF Y2 = p2.y;                                            //                   < 2
      const FF t00 = X1 * X2;                                        // t00 ← X1 · X2     < 2
      const FF t01 = Y1 * Y2;                                        // t01 ← Y1 · Y2     < 2
      const FF t02 = Z1;                                             // t02 ← Z1          < 2
      const FF t03 = X1 + Y1;                                        // t03 ← X1 + Y1     < 4
      const FF t04 = X2 + Y2;                                        // t04 ← X2 + Y2     < 4
      const FF t05 = t03 * t04;                                      // t03 ← t03 · t04   < 3
      const FF t06 = t00 + t01;                                      // t06 ← t00 + t01   < 4
      const FF t07 = t05 - t06;                                      // t05 ← t05 − t06   < 2
      const FF t08 = Y1 + Z1;                                        // t08 ← Y1 + Z1     < 4
      const FF t09 = Y2 + FF::one();                                 // t09 ← Y2 + 1      < 4
      const FF t10 = t08 * t09;                                      // t10 ← t08 · t09   < 3
      const FF t11 = t01 + t02;                                      // t11 ← t01 + t02   < 4
      const FF t12 = t10 - t11;                                      // t12 ← t10 − t11   < 2
      const FF t13 = X1 + Z1;                                        // t13 ← X1 + Z1     < 4
      const FF t14 = X2 + FF::one();                                 // t14 ← X2 + 1      < 4
      const FF t15 = t13 * t14;                                      // t15 ← t13 · t14   < 3
      const FF t16 = t00 + t02;                                      // t16 ← t00 + t02   < 4
      const FF t17 = t15 - t16;                                      // t17 ← t15 − t16   < 2
      const FF t18 = t00 + t00;                                      // t18 ← t00 + t00   < 2
      const FF t19 = t18 + t00;                                      // t19 ← t18 + t00   < 2
      const FF t20 = FF::template mul_weierstrass_b<Gen, true>(t02); // t20 ← b3 · t02    < 2
      const FF t21 = t01 + t20;                                      // t21 ← t01 + t20   < 2
      const FF t22 = t01 - t20;                                      // t22 ← t01 − t20   < 2
      const FF t23 = FF::template mul_weierstrass_b<Gen, true>(t17); // t23 ← b3 · t17    < 2
      const auto t24 = FF::mul_wide(t12, t23);                       // t24 ← t12 · t23   < 2
      const auto t25 = FF::mul_wide(t07, t22);                       // t25 ← t07 · t22   < 2
      const FF X3 = FF::reduce(t25 - t24);                           // X3 ← t25 − t24    < 2
      const auto t27 = FF::mul_wide(t23, t19);                       // t27 ← t23 · t19   < 2
      const auto t28 = FF::mul_wide(t22, t21);                       // t28 ← t22 · t21   < 2
      const FF Y3 = FF::reduce(t28 + t27);                           // Y3 ← t28 + t27    < 2
      const auto t30 = FF::mul_wide(t19, t07);                       // t30 ← t19 · t07   < 2
      const auto t31 = FF::mul_wide(t21, t12);                       // t31 ← t21 · t12   < 2
      const FF Z3 = FF::reduce(t31 + t30);                           // Z3 ← t31 + t30    < 2
      p1.x = X3;
      p1.y = Y3;
      p1.z = Z3;
    }
  }

  static HOST_DEVICE void accum_prj_prj(Projective &p1, const Projective &p2) {
    if constexpr (std::is_same_v<Gen, bn254::G1>) {
      const mp_256_t &X1 = reinterpret_cast<mp_256_t &>(p1.x);
      const mp_256_t &Y1 = reinterpret_cast<mp_256_t &>(p1.y);
      const mp_256_t &Z1 = reinterpret_cast<mp_256_t &>(p1.z);
      const mp_256_t &X2 = reinterpret_cast<const mp_256_t &>(p2.x);
      const mp_256_t &Y2 = reinterpret_cast<const mp_256_t &>(p2.y);
      const mp_256_t &Z2 = reinterpret_cast<const mp_256_t &>(p2.z);
      const mp_256_t t00 = bn254_mont_t::mul(X1, X2);
      const mp_256_t t01 = bn254_mont_t::mul(Y1, Y2);
      const mp_256_t t02 = bn254_mont_t::mul(Z1, Z2);
      const mp_256_t t03 = bn254_mont_t::add(X1, Y1);
      const mp_256_t t04 = bn254_mont_t::add(X2, Y2);
      const mp_256_t t05 = bn254_mont_t::mul(t03, t04);
      const mp_256_t t06 = bn254_mont_t::add_red(t00, t01);
      const mp_256_t t07 = bn254_mont_t::sub_red(t05, t06);
      const mp_256_t t08 = bn254_mont_t::add(Y1, Z1);
      const mp_256_t t09 = bn254_mont_t::add(Y2, Z2);
      const mp_256_t t10 = bn254_mont_t::mul(t08, t09);
      const mp_256_t t11 = bn254_mont_t::add_red(t01, t02);
      const mp_256_t t12 = bn254_mont_t::sub_red(t10, t11);
      const mp_256_t t13 = bn254_mont_t::add(X1, Z1);
      const mp_256_t t14 = bn254_mont_t::add(X2, Z2);
      const mp_256_t t15 = bn254_mont_t::mul(t13, t14);
      const mp_256_t t16 = bn254_mont_t::add_red(t00, t02);
      const mp_256_t t17 = bn254_mont_t::sub_red(t15, t16);
      const mp_256_t t18 = bn254_mont_t::add_red(t00, t00);
      const mp_256_t t19 = bn254_mont_t::add_red(t18, t00);
      const mp_256_t t20 = mul_b3(t02);
      const mp_256_t t21 = bn254_mont_t::add_red(t01, t20);
      const mp_256_t t22 = bn254_mont_t::sub_red(t01, t20);
      const mp_256_t t23 = mul_b3(t17);
      const auto t24 = bn254_mont_t::schoolbook_mul(t12, t23);
      const auto t25 = bn254_mont_t::schoolbook_mul(t07, t22);
      const mp_256_t X3 = bn254_mont_t::reduce(bn254_mont_t::sub_red(t25, t24));
      const auto t27 = bn254_mont_t::schoolbook_mul(t23, t19);
      const auto t28 = bn254_mont_t::schoolbook_mul(t22, t21);
      const mp_256_t Y3 = bn254_mont_t::reduce(bn254_mont_t::add(t28, t27));
      const auto t30 = bn254_mont_t::schoolbook_mul(t19, t07);
      const auto t31 = bn254_mont_t::schoolbook_mul(t21, t12);
      const mp_256_t Z3 = bn254_mont_t::reduce(bn254_mont_t::add(t31, t30));
      // auto end = get_clock_counter();
      // uint64_t ns = end - start;
      // prj_time.fetch_add(ns, std::memory_order_relaxed);
      p1.x = reinterpret_cast<const FF &>(X3);
      p1.y = reinterpret_cast<const FF &>(Y3);
      p1.z = reinterpret_cast<const FF &>(Z3);
    } else if constexpr (std::is_same_v<Gen, bn254::G2>) {
      const c_bn254_t X1 = reinterpret_cast<const c_bn254_t &>(p1.x);    //                   < 2
      const c_bn254_t Y1 = reinterpret_cast<const c_bn254_t &>(p1.y);    //                   < 2
      const c_bn254_t Z1 = reinterpret_cast<const c_bn254_t &>(p1.z);    //                   < 2
      const c_bn254_t X2 = reinterpret_cast<const c_bn254_t &>(p2.x);    //                   < 2
      const c_bn254_t Y2 = reinterpret_cast<const c_bn254_t &>(p2.y);    //                   < 2
      const c_bn254_t Z2 = reinterpret_cast<const c_bn254_t &>(p2.z);    //                   < 2
      const c_bn254_t t00 = mont_mul(X1, X2);                 // t00 ← X1 · X2     < 2
      const c_bn254_t t01 = mont_mul(Y1, Y2);                 // t01 ← Y1 · Y2     < 2
      const c_bn254_t t02 = mont_mul(Z1, Z2);                 // t02 ← Z1 · Z2     < 2
      const c_bn254_t t03 = add(X1, Y1);                      // t03 ← X1 + Y1     < 4
      const c_bn254_t t04 = add(X2, Y2);                      // t04 ← X2 + Y2     < 4
      const c_bn254_t t05 = mont_mul(t03, t04);               // t03 ← t03 · t04   < 3
      const c_bn254_t t06 = add(t00, t01);                    // t06 ← t00 + t01   < 4
      const c_bn254_t t07 = sub(t05, t06);                    // t05 ← t05 − t06   < 2
      const c_bn254_t t08 = add(Y1, Z1);                      // t08 ← Y1 + Z1     < 4
      const c_bn254_t t09 = add(Y2, Z2);                      // t09 ← Y2 + Z2     < 4
      const c_bn254_t t10 = mont_mul(t08, t09);               // t10 ← t08 · t09   < 3
      const c_bn254_t t11 = add(t01, t02);                    // t11 ← t01 + t02   < 4
      const c_bn254_t t12 = sub(t10, t11);                    // t12 ← t10 − t11   < 2
      const c_bn254_t t13 = add(X1, Z1);                      // t13 ← X1 + Z1     < 4
      const c_bn254_t t14 = add(X2, Z2);                      // t14 ← X2 + Z2     < 4
      const c_bn254_t t15 = mont_mul(t13, t14);               // t15 ← t13 · t14   < 3
      const c_bn254_t t16 = add(t00, t02);                    // t16 ← t00 + t02   < 4
      const c_bn254_t t17 = sub(t15, t16);                    // t17 ← t15 − t16   < 2
      const c_bn254_t t18 = add(t00, t00);                    // t18 ← t00 + t00   < 2
      const c_bn254_t t19 = add(t18, t00);                    // t19 ← t18 + t00   < 2
      const c_bn254_t t20 = mul_b3(t02);                      // t20 ← b3 · t02    < 2
      const c_bn254_t t21 = add(t01, t20);                    // t21 ← t01 + t20   < 2
      const c_bn254_t t22 = sub(t01, t20);                    // t22 ← t01 − t20   < 2
      const c_bn254_t t23 = mul_b3(t17);                      // t23 ← b3 · t17    < 2
      const auto t24 = mul_wide(t12, t23);                    // t24 ← t12 · t23   < 2
      const auto t25 = mul_wide(t07, t22);                    // t25 ← t07 · t22   < 2
      const auto X3 = reduce(sub(t25, t24));                  // X3 ← t25 − t24    < 2
      const auto t27 = mul_wide(t23, t19);                    // t27 ← t23 · t19   < 2
      const auto t28 = mul_wide(t22, t21);                    // t28 ← t22 · t21   < 2
      const auto Y3 = reduce(add_no_red(t28, t27));                  // Y3 ← t28 + t27    < 2
      const auto t30 = mul_wide(t19, t07);                    // t30 ← t19 · t07   < 2
      const auto t31 = mul_wide(t21, t12);                    // t31 ← t21 · t12   < 2
      const auto Z3 = reduce(add_no_red(t31, t30));                  // Z3 ← t31 + t30    < 2
      p1.x = reinterpret_cast<const FF &>(X3);
      p1.y = reinterpret_cast<const FF &>(Y3);
      p1.z = reinterpret_cast<const FF &>(Z3);
    } else {
      const FF X1 = p1.x;                                            //                   < 2
      const FF Y1 = p1.y;                                            //                   < 2
      const FF Z1 = p1.z;                                            //                   < 2
      const FF X2 = p2.x;                                            //                   < 2
      const FF Y2 = p2.y;                                            //                   < 2
      const FF Z2 = p2.z;                                            //                   < 2
      const FF t00 = X1 * X2;                                        // t00 ← X1 · X2     < 2
      const FF t01 = Y1 * Y2;                                        // t01 ← Y1 · Y2     < 2
      const FF t02 = Z1 * Z2;                                        // t02 ← Z1 · Z2     < 2
      const FF t03 = X1 + Y1;                                        // t03 ← X1 + Y1     < 4
      const FF t04 = X2 + Y2;                                        // t04 ← X2 + Y2     < 4
      const FF t05 = t03 * t04;                                      // t03 ← t03 · t04   < 3
      const FF t06 = t00 + t01;                                      // t06 ← t00 + t01   < 4
      const FF t07 = t05 - t06;                                      // t05 ← t05 − t06   < 2
      const FF t08 = Y1 + Z1;                                        // t08 ← Y1 + Z1     < 4
      const FF t09 = Y2 + Z2;                                        // t09 ← Y2 + Z2     < 4
      const FF t10 = t08 * t09;                                      // t10 ← t08 · t09   < 3
      const FF t11 = t01 + t02;                                      // t11 ← t01 + t02   < 4
      const FF t12 = t10 - t11;                                      // t12 ← t10 − t11   < 2
      const FF t13 = X1 + Z1;                                        // t13 ← X1 + Z1     < 4
      const FF t14 = X2 + Z2;                                        // t14 ← X2 + Z2     < 4
      const FF t15 = t13 * t14;                                      // t15 ← t13 · t14   < 3
      const FF t16 = t00 + t02;                                      // t16 ← t00 + t02   < 4
      const FF t17 = t15 - t16;                                      // t17 ← t15 − t16   < 2
      const FF t18 = t00 + t00;                                      // t18 ← t00 + t00   < 2
      const FF t19 = t18 + t00;                                      // t19 ← t18 + t00   < 2
      const FF t20 = FF::template mul_weierstrass_b<Gen, true>(t02); // t20 ← b3 · t02    < 2
      const FF t21 = t01 + t20;                                      // t21 ← t01 + t20   < 2
      const FF t22 = t01 - t20;                                      // t22 ← t01 − t20   < 2
      const FF t23 = FF::template mul_weierstrass_b<Gen, true>(t17); // t23 ← b3 · t17    < 2
      const auto t24 = FF::mul_wide(t12, t23);                       // t24 ← t12 · t23   < 2
      const auto t25 = FF::mul_wide(t07, t22);                       // t25 ← t07 · t22   < 2
      const FF X3 = FF::reduce(t25 - t24);                           // X3 ← t25 − t24    < 2
      const auto t27 = FF::mul_wide(t23, t19);                       // t27 ← t23 · t19   < 2
      const auto t28 = FF::mul_wide(t22, t21);                       // t28 ← t22 · t21   < 2
      const FF Y3 = FF::reduce(t28 + t27);                           // Y3 ← t28 + t27    < 2
      const auto t30 = FF::mul_wide(t19, t07);                       // t30 ← t19 · t07   < 2
      const auto t31 = FF::mul_wide(t21, t12);                       // t31 ← t21 · t12   < 2
      const FF Z3 = FF::reduce(t31 + t30);                           // Z3 ← t31 + t30    < 2
      // auto end = get_clock_counter();
      // uint64_t ns = end - start;
      // prj_time.fetch_add(ns, std::memory_order_relaxed);
      p1.x = X3;
      p1.y = Y3;
      p1.z = Z3;
    }
  }
  // ---------------------------------------------------------------------------------------------------

  static HOST_DEVICE Projective dbl(const Projective& point)
  {
    const FF X = point.x;
    const FF Y = point.y;
    const FF Z = point.z;

    // TODO: Change to efficient dbl once implemented for field.cuh
    FF t0 = FF::sqr(Y);                                 // 1. t0 ← Y · Y
    FF Z3 = t0 + t0;                                    // 2. Z3 ← t0 + t0
    Z3 = Z3 + Z3;                                       // 3. Z3 ← Z3 + Z3
    Z3 = Z3 + Z3;                                       // 4. Z3 ← Z3 + Z3
    FF t1 = Y * Z;                                      // 5. t1 ← Y · Z
    FF t2 = FF::sqr(Z);                                 // 6. t2 ← Z · Z
    t2 = FF::template mul_weierstrass_b<Gen, true>(t2); // 7. t2 ← b3 · t2
    FF X3 = t2 * Z3;                                    // 8. X3 ← t2 · Z3
    FF Y3 = t0 + t2;                                    // 9. Y3 ← t0 + t2
    Z3 = t1 * Z3;                                       // 10. Z3 ← t1 · Z3
    t1 = t2 + t2;                                       // 11. t1 ← t2 + t2
    t2 = t1 + t2;                                       // 12. t2 ← t1 + t2
    t0 = t0 - t2;                                       // 13. t0 ← t0 − t2
    Y3 = t0 * Y3;                                       // 14. Y3 ← t0 · Y3
    Y3 = X3 + Y3;                                       // 15. Y3 ← X3 + Y3
    t1 = X * Y;                                         // 16. t1 ← X · Y
    X3 = t0 * t1;                                       // 17. X3 ← t0 · t1
    X3 = X3 + X3;                                       // 18. X3 ← X3 + X3
    return {X3, Y3, Z3};
  }

  friend HOST_DEVICE Projective operator+(Projective p1, const Projective& p2)
  {
    const FF X1 = p1.x;                                            //                   < 2
    const FF Y1 = p1.y;                                            //                   < 2
    const FF Z1 = p1.z;                                            //                   < 2
    const FF X2 = p2.x;                                            //                   < 2
    const FF Y2 = p2.y;                                            //                   < 2
    const FF Z2 = p2.z;                                            //                   < 2
    const FF t00 = X1 * X2;                                        // t00 ← X1 · X2     < 2
    const FF t01 = Y1 * Y2;                                        // t01 ← Y1 · Y2     < 2
    const FF t02 = Z1 * Z2;                                        // t02 ← Z1 · Z2     < 2
    const FF t03 = X1 + Y1;                                        // t03 ← X1 + Y1     < 4
    const FF t04 = X2 + Y2;                                        // t04 ← X2 + Y2     < 4
    const FF t05 = t03 * t04;                                      // t03 ← t03 · t04   < 3
    const FF t06 = t00 + t01;                                      // t06 ← t00 + t01   < 4
    const FF t07 = t05 - t06;                                      // t05 ← t05 − t06   < 2
    const FF t08 = Y1 + Z1;                                        // t08 ← Y1 + Z1     < 4
    const FF t09 = Y2 + Z2;                                        // t09 ← Y2 + Z2     < 4
    const FF t10 = t08 * t09;                                      // t10 ← t08 · t09   < 3
    const FF t11 = t01 + t02;                                      // t11 ← t01 + t02   < 4
    const FF t12 = t10 - t11;                                      // t12 ← t10 − t11   < 2
    const FF t13 = X1 + Z1;                                        // t13 ← X1 + Z1     < 4
    const FF t14 = X2 + Z2;                                        // t14 ← X2 + Z2     < 4
    const FF t15 = t13 * t14;                                      // t15 ← t13 · t14   < 3
    const FF t16 = t00 + t02;                                      // t16 ← t00 + t02   < 4
    const FF t17 = t15 - t16;                                      // t17 ← t15 − t16   < 2
    const FF t18 = t00 + t00;                                      // t18 ← t00 + t00   < 2
    const FF t19 = t18 + t00;                                      // t19 ← t18 + t00   < 2
    const FF t20 = FF::template mul_weierstrass_b<Gen, true>(t02); // t20 ← b3 · t02    < 2
    const FF t21 = t01 + t20;                                      // t21 ← t01 + t20   < 2
    const FF t22 = t01 - t20;                                      // t22 ← t01 − t20   < 2
    const FF t23 = FF::template mul_weierstrass_b<Gen, true>(t17); // t23 ← b3 · t17    < 2
    const auto t24 = FF::mul_wide(t12, t23);                       // t24 ← t12 · t23   < 2
    const auto t25 = FF::mul_wide(t07, t22);                       // t25 ← t07 · t22   < 2
    const FF X3 = FF::reduce(t25 - t24);                           // X3 ← t25 − t24    < 2
    const auto t27 = FF::mul_wide(t23, t19);                       // t27 ← t23 · t19   < 2
    const auto t28 = FF::mul_wide(t22, t21);                       // t28 ← t22 · t21   < 2
    const FF Y3 = FF::reduce(t28 + t27);                           // Y3 ← t28 + t27    < 2
    const auto t30 = FF::mul_wide(t19, t07);                       // t30 ← t19 · t07   < 2
    const auto t31 = FF::mul_wide(t21, t12);                       // t31 ← t21 · t12   < 2
    const FF Z3 = FF::reduce(t31 + t30);                           // Z3 ← t31 + t30    < 2
    return {X3, Y3, Z3};
  }

  friend HOST_DEVICE_INLINE Projective operator-(Projective p1, const Projective& p2) { return p1 + neg(p2); }

  friend HOST_DEVICE Projective operator+(Projective p1, const Affine<FF>& p2)
  {
    const FF X1 = p1.x;                                            //                   < 2
    const FF Y1 = p1.y;                                            //                   < 2
    const FF Z1 = p1.z;                                            //                   < 2
    const FF X2 = p2.x;                                            //                   < 2
    const FF Y2 = p2.y;                                            //                   < 2
    const FF t00 = X1 * X2;                                        // t00 ← X1 · X2     < 2
    const FF t01 = Y1 * Y2;                                        // t01 ← Y1 · Y2     < 2
    const FF t02 = Z1;                                             // t02 ← Z1          < 2
    const FF t03 = X1 + Y1;                                        // t03 ← X1 + Y1     < 4
    const FF t04 = X2 + Y2;                                        // t04 ← X2 + Y2     < 4
    const FF t05 = t03 * t04;                                      // t03 ← t03 · t04   < 3
    const FF t06 = t00 + t01;                                      // t06 ← t00 + t01   < 4
    const FF t07 = t05 - t06;                                      // t05 ← t05 − t06   < 2
    const FF t08 = Y1 + Z1;                                        // t08 ← Y1 + Z1     < 4
    const FF t09 = Y2 + FF::one();                                 // t09 ← Y2 + 1      < 4
    const FF t10 = t08 * t09;                                      // t10 ← t08 · t09   < 3
    const FF t11 = t01 + t02;                                      // t11 ← t01 + t02   < 4
    const FF t12 = t10 - t11;                                      // t12 ← t10 − t11   < 2
    const FF t13 = X1 + Z1;                                        // t13 ← X1 + Z1     < 4
    const FF t14 = X2 + FF::one();                                 // t14 ← X2 + 1      < 4
    const FF t15 = t13 * t14;                                      // t15 ← t13 · t14   < 3
    const FF t16 = t00 + t02;                                      // t16 ← t00 + t02   < 4
    const FF t17 = t15 - t16;                                      // t17 ← t15 − t16   < 2
    const FF t18 = t00 + t00;                                      // t18 ← t00 + t00   < 2
    const FF t19 = t18 + t00;                                      // t19 ← t18 + t00   < 2
    const FF t20 = FF::template mul_weierstrass_b<Gen, true>(t02); // t20 ← b3 · t02    < 2
    const FF t21 = t01 + t20;                                      // t21 ← t01 + t20   < 2
    const FF t22 = t01 - t20;                                      // t22 ← t01 − t20   < 2
    const FF t23 = FF::template mul_weierstrass_b<Gen, true>(t17); // t23 ← b3 · t17    < 2
    const auto t24 = FF::mul_wide(t12, t23);                       // t24 ← t12 · t23   < 2
    const auto t25 = FF::mul_wide(t07, t22);                       // t25 ← t07 · t22   < 2
    const FF X3 = FF::reduce(t25 - t24);                           // X3 ← t25 − t24    < 2
    const auto t27 = FF::mul_wide(t23, t19);                       // t27 ← t23 · t19   < 2
    const auto t28 = FF::mul_wide(t22, t21);                       // t28 ← t22 · t21   < 2
    const FF Y3 = FF::reduce(t28 + t27);                           // Y3 ← t28 + t27    < 2
    const auto t30 = FF::mul_wide(t19, t07);                       // t30 ← t19 · t07   < 2
    const auto t31 = FF::mul_wide(t21, t12);                       // t31 ← t21 · t12   < 2
    const FF Z3 = FF::reduce(t31 + t30);                           // Z3 ← t31 + t30    < 2
    return {X3, Y3, Z3};
  }

  friend HOST_DEVICE_INLINE Projective operator-(Projective p1, const Affine<FF>& p2)
  {
    return p1 + Affine<FF>::neg(p2);
  }

  friend HOST_DEVICE Projective operator*(SCALAR_FF scalar, const Projective& point)
  {
    // Precompute points: P, 2P, ..., (2^window_size - 1)P
    constexpr unsigned window_size =
      4; // 4 seems fastest. Optimum is minimizing EC add and depends on the field size. for 256b it's 4.
    constexpr unsigned table_size = (1 << window_size) - 1; // 2^window_size-1
    std::array<Projective, table_size> table;
    table[0] = point;
    for (int i = 1; i < table_size; ++i) {
      table[i] = table[i - 1] + point; // Compute (i+1)P
    }

    Projective res = zero();

    constexpr int nof_windows = (SCALAR_FF::NBITS + window_size - 1) / window_size;
    bool res_is_not_zero = false;
    for (int w = nof_windows - 1; w >= 0; w -= 1) {
      // Extract the next window_size bits from the scalar
      unsigned window = scalar.get_scalar_digit(w, window_size);

      // Double the result window_size times
      for (int j = 0; res_is_not_zero && j < window_size; ++j) {
        res = dbl(res); // Point doubling
      }

      // Add the precomputed value if window is not zero
      if (window != 0) {
        res = res + table[window - 1]; // Add the precomputed point
        res_is_not_zero = true;
      }
    }
    return res;
  }

  friend HOST_DEVICE_INLINE Projective operator*(const Projective& point, SCALAR_FF scalar) { return scalar * point; }

  friend HOST_DEVICE_INLINE bool operator==(const Projective& p1, const Projective& p2)
  {
    return (p1.x * p2.z == p2.x * p1.z) && (p1.y * p2.z == p2.y * p1.z);
  }

  friend HOST_DEVICE_INLINE bool operator!=(const Projective& p1, const Projective& p2) { return !(p1 == p2); }

  friend HOST_INLINE std::ostream& operator<<(std::ostream& os, const Projective& point)
  {
    os << "Point { x: " << point.x << "; y: " << point.y << "; z: " << point.z << " }";
    return os;
  }

  static HOST_DEVICE_INLINE bool is_zero(const Projective& point)
  {
    return point.x == FF::zero() && point.y != FF::zero() && point.z == FF::zero();
  }

  static HOST_DEVICE_INLINE bool is_on_curve(const Projective& point)
  {
    if (is_zero(point)) return true;
    bool eq_holds =
      (FF::template mul_weierstrass_b<Gen>(FF::sqr(point.z) * point.z) + FF::sqr(point.x) * point.x ==
       point.z * FF::sqr(point.y));
    return point.z != FF::zero() && eq_holds;
  }

  static HOST_INLINE Affine<FF> rand_host_affine() { return to_affine(rand_host()); }

  static HOST_INLINE Projective rand_host()
  {
    SCALAR_FF rand_scalar = SCALAR_FF::rand_host();
    return rand_scalar * generator();
  }

  static void rand_host_many(Projective* out, int size)
  {
    for (int i = 0; i < size; i++)
      out[i] = (i % size < 100) ? rand_host() : out[i - 100];
  }

  static void rand_host_many(Affine<FF>* out, int size)
  {
    for (int i = 0; i < size; i++)
      out[i] = (i % size < 100) ? to_affine(rand_host()) : out[i - 100];
  }
};

#ifdef __CUDACC__
template <typename FF, class SCALAR_FF, typename Gen>
struct SharedMemory<Projective<FF, SCALAR_FF, Gen>> {
  __device__ Projective<FF, SCALAR_FF, Gen>* getPointer()
  {
    extern __shared__ Projective<FF, SCALAR_FF, Gen> s_projective_[];
    return s_projective_;
  }
};
#endif // __CUDACC__
