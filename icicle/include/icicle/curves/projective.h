#pragma once

#ifdef __CUDACC__
  #include "gpu-utils/sharedmem.h"
#endif // __CUDACC__

#include "icicle/utils/modifiers.h"
#include "icicle/curves/affine.h"
#include <array>

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

  static HOST_DEVICE_INLINE Projective from_affine(const Affine<FF>& point)
  {
    return point.is_zero() ? zero() : Projective{point.x, point.y, FF::one()};
  }

  static HOST_DEVICE_INLINE Projective generator() { return {Gen::gen_x, Gen::gen_y, FF::one()}; }

  static HOST_INLINE Affine<FF> rand_host_affine() { return rand_host().to_affine(); }

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
      out[i] = (i % size < 100) ? rand_host().to_affine() : out[i - 100];
  }

  HOST_DEVICE_INLINE Affine<FF> to_affine()
  {
    FF denom = z.inverse();
    return {x * denom, y * denom};
  }

  HOST_DEVICE_INLINE Projective to_montgomery() const
  {
    return {x.to_montgomery(), y.to_montgomery(), z.to_montgomery()};
  }

  HOST_DEVICE_INLINE Projective from_montgomery() const
  {
    return {x.from_montgomery(), y.from_montgomery(), z.from_montgomery()};
  }

  HOST_DEVICE_INLINE Projective neg() const { return {x, y.neg(), z}; }

  HOST_DEVICE Projective dbl() const
  {
    const FF X = x;
    const FF Y = y;
    const FF Z = z;

    // TODO: Change to efficient dbl once implemented for field.cuh
    FF t0 = Y.sqr();                                    // 1. t0 ← Y · Y
    FF Z3 = t0 + t0;                                    // 2. Z3 ← t0 + t0
    Z3 = Z3 + Z3;                                       // 3. Z3 ← Z3 + Z3
    Z3 = Z3 + Z3;                                       // 4. Z3 ← Z3 + Z3
    FF t1 = Y * Z;                                      // 5. t1 ← Y · Z
    FF t2 = Z.sqr();                                    // 6. t2 ← Z · Z
    t2 = FF::template mul_weierstrass_b<Gen>(t2);      // 7. t2 ← b3 · t2
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

  HOST_DEVICE Projective operator+(const Projective& p2) const
  {
    const FF X1 = x;                                               //                   < 2
    const FF Y1 = y;                                               //                   < 2
    const FF Z1 = z;                                               //                   < 2
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
    const auto t24 = t12.mul_wide(t23);                            // t24 ← t12 · t23   < 2
    const auto t25 = t07.mul_wide(t22);                            // t25 ← t07 · t22   < 2
    const FF X3 = (t25 - t24).reduce();                            // X3 ← t25 − t24    < 2
    const auto t27 = t23.mul_wide(t19);                            // t27 ← t23 · t19   < 2
    const auto t28 = t22.mul_wide(t21);                            // t28 ← t22 · t21   < 2
    const FF Y3 = (t28 + t27).reduce();                            // Y3 ← t28 + t27    < 2
    const auto t30 = t19.mul_wide(t07);                            // t30 ← t19 · t07   < 2
    const auto t31 = t21.mul_wide(t12);                            // t31 ← t21 · t12   < 2
    const FF Z3 = (t31 + t30).reduce();                            // Z3 ← t31 + t30    < 2
    return {X3, Y3, Z3};
  }

  HOST_DEVICE_INLINE Projective operator-(const Projective& p2) const { return *this + p2.neg(); }

  HOST_DEVICE Projective operator+(const Affine<FF>& p2) const
  {
    const FF X1 = x;                                               //                   < 2
    const FF Y1 = y;                                               //                   < 2
    const FF Z1 = z;                                               //                   < 2
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
    const auto t24 = t12.mul_wide(t23);                            // t24 ← t12 · t23   < 2
    const auto t25 = t07.mul_wide(t22);                            // t25 ← t07 · t22   < 2
    const FF X3 = (t25 - t24).reduce();                            // X3 ← t25 − t24    < 2
    const auto t27 = t23.mul_wide(t19);                            // t27 ← t23 · t19   < 2
    const auto t28 = t22.mul_wide(t21);                            // t28 ← t22 · t21   < 2
    const FF Y3 = (t28 + t27).reduce();                            // Y3 ← t28 + t27    < 2
    const auto t30 = t19.mul_wide(t07);                            // t30 ← t19 · t07   < 2
    const auto t31 = t21.mul_wide(t12);                            // t31 ← t21 · t12   < 2
    const FF Z3 = (t31 + t30).reduce();                            // Z3 ← t31 + t30    < 2
    return {X3, Y3, Z3};
  }

  HOST_DEVICE_INLINE Projective operator-(const Affine<FF>& p2) const { return *this + p2.neg(); }

  HOST_DEVICE_INLINE Projective operator*(SCALAR_FF scalar) const
  {
    // Precompute points: P, 2P, ..., (2^window_size - 1)P
    constexpr unsigned window_size =
      4; // 4 seems fastest. Optimum is minimizing EC add and depends on the field size. for 256b it's 4.
    constexpr unsigned table_size = (1 << window_size) - 1; // 2^window_size-1
    std::array<Projective, table_size> table;
    table[0] = *this;
    for (int i = 1; i < table_size; ++i) {
      table[i] = table[i - 1] + *this; // Compute (i+1)P
    }

    Projective res = zero();

    constexpr int nof_windows = (SCALAR_FF::NBITS + window_size - 1) / window_size;
    bool res_is_not_zero = false;
    for (int w = nof_windows - 1; w >= 0; w -= 1) {
      // Extract the next window_size bits from the scalar
      unsigned window = scalar.get_scalar_digit(w, window_size);

      // Double the result window_size times
      for (int j = 0; res_is_not_zero && j < window_size; ++j) {
        res = res.dbl(); // Point doubling
      }

      // Add the precomputed value if window is not zero
      if (window != 0) {
        res = res + table[window - 1]; // Add the precomputed point
        res_is_not_zero = true;
      }
    }
    return res;
  }

  friend HOST_DEVICE Projective operator*(SCALAR_FF scalar, const Projective& point) { return point * scalar; }

  HOST_DEVICE_INLINE bool operator==(const Projective& p2) const
  {
    return (x * p2.z == p2.x * z) && (y * p2.z == p2.y * z);
  }

  HOST_DEVICE_INLINE bool operator!=(const Projective& p2) const { return !(*this == p2); }

  friend HOST_INLINE std::ostream& operator<<(std::ostream& os, const Projective& point)
  {
    os << "Point { x: " << point.x << "; y: " << point.y << "; z: " << point.z << " }";
    return os;
  }

  HOST_DEVICE_INLINE bool is_zero() const { return x == FF::zero() && y != FF::zero() && z == FF::zero(); }

  HOST_DEVICE_INLINE bool is_on_curve() const
  {
    if (is_zero()) return true;
    bool eq_holds = (FF::template mul_weierstrass_b<Gen>(z.sqr() * z) + x.sqr() * x == z * y.sqr());
    return z != FF::zero() && eq_holds;
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
