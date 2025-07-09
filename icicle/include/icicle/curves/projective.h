#pragma once

#ifdef __CUDACC__
  #include "gpu-utils/sharedmem.h"
#endif // __CUDACC__

#include "icicle/utils/modifiers.h"
#include "icicle/curves/affine.h"
#include <array>

template <typename BaseField, class ScalarField, typename Gen>
class Projective
{
  friend Affine<BaseField>;

public:
  typedef Affine<BaseField> Aff;
  typedef ScalarField Scalar;

  static constexpr unsigned SCALAR_FIELD_NBITS = ScalarField::NBITS;
  static constexpr unsigned BASE_FIELD_NBITS = BaseField::NBITS;

  BaseField x;
  BaseField y;
  BaseField z;
  static HOST_DEVICE_INLINE Projective zero() { return {BaseField::zero(), BaseField::one(), BaseField::zero()}; }

  static HOST_DEVICE_INLINE Projective from_affine(const Affine<BaseField>& point)
  {
    return point.is_zero() ? zero() : Projective{point.x, point.y, BaseField::one()};
  }

  static HOST_DEVICE_INLINE Projective generator() { return {Gen::gen_x, Gen::gen_y, BaseField::one()}; }

  static HOST_INLINE Affine<BaseField> rand_host_affine() { return rand_host().to_affine(); }

  static Projective rand_host()
  {
    ScalarField rand_scalar = ScalarField::rand_host();
    return rand_scalar * generator();
  }

  static void rand_host_many(Projective* out, int size)
  {
    for (int i = 0; i < size; i++)
      out[i] = (i % size < 100) ? rand_host() : out[i - 100];
  }

  static void rand_host_many(Affine<BaseField>* out, int size)
  {
    for (int i = 0; i < size; i++)
      out[i] = (i % size < 100) ? rand_host().to_affine() : out[i - 100];
  }

  HOST_DEVICE_INLINE Affine<BaseField> to_affine()
  {
    BaseField denom = z.inverse();
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
    const BaseField X = x;
    const BaseField Y = y;
    const BaseField Z = z;

    // TODO: Change to efficient dbl once implemented for field.cuh
    BaseField t0 = Y.sqr();                                    // 1. t0 ← Y · Y
    BaseField Z3 = t0 + t0;                                    // 2. Z3 ← t0 + t0
    Z3 = Z3 + Z3;                                              // 3. Z3 ← Z3 + Z3
    Z3 = Z3 + Z3;                                              // 4. Z3 ← Z3 + Z3
    BaseField t1 = Y * Z;                                      // 5. t1 ← Y · Z
    BaseField t2 = Z.sqr();                                    // 6. t2 ← Z · Z
    t2 = BaseField::template mul_weierstrass_b<Gen, true>(t2); // 7. t2 ← b3 · t2
    BaseField X3 = t2 * Z3;                                    // 8. X3 ← t2 · Z3
    BaseField Y3 = t0 + t2;                                    // 9. Y3 ← t0 + t2
    Z3 = t1 * Z3;                                              // 10. Z3 ← t1 · Z3
    t1 = t2 + t2;                                              // 11. t1 ← t2 + t2
    t2 = t1 + t2;                                              // 12. t2 ← t1 + t2
    t0 = t0 - t2;                                              // 13. t0 ← t0 − t2
    Y3 = t0 * Y3;                                              // 14. Y3 ← t0 · Y3
    Y3 = X3 + Y3;                                              // 15. Y3 ← X3 + Y3
    t1 = X * Y;                                                // 16. t1 ← X · Y
    X3 = t0 * t1;                                              // 17. X3 ← t0 · t1
    X3 = X3 + X3;                                              // 18. X3 ← X3 + X3
    return {X3, Y3, Z3};
  }

  HOST_DEVICE Projective operator+(const Projective& p2) const
  {
    const BaseField X1 = x;                                                      //                   < 2
    const BaseField Y1 = y;                                                      //                   < 2
    const BaseField Z1 = z;                                                      //                   < 2
    const BaseField X2 = p2.x;                                                   //                   < 2
    const BaseField Y2 = p2.y;                                                   //                   < 2
    const BaseField Z2 = p2.z;                                                   //                   < 2
    const BaseField t00 = X1 * X2;                                               // t00 ← X1 · X2     < 2
    const BaseField t01 = Y1 * Y2;                                               // t01 ← Y1 · Y2     < 2
    const BaseField t02 = Z1 * Z2;                                               // t02 ← Z1 · Z2     < 2
    const BaseField t03 = X1 + Y1;                                               // t03 ← X1 + Y1     < 4
    const BaseField t04 = X2 + Y2;                                               // t04 ← X2 + Y2     < 4
    const BaseField t05 = t03 * t04;                                             // t03 ← t03 · t04   < 3
    const BaseField t06 = t00 + t01;                                             // t06 ← t00 + t01   < 4
    const BaseField t07 = t05 - t06;                                             // t05 ← t05 − t06   < 2
    const BaseField t08 = Y1 + Z1;                                               // t08 ← Y1 + Z1     < 4
    const BaseField t09 = Y2 + Z2;                                               // t09 ← Y2 + Z2     < 4
    const BaseField t10 = t08 * t09;                                             // t10 ← t08 · t09   < 3
    const BaseField t11 = t01 + t02;                                             // t11 ← t01 + t02   < 4
    const BaseField t12 = t10 - t11;                                             // t12 ← t10 − t11   < 2
    const BaseField t13 = X1 + Z1;                                               // t13 ← X1 + Z1     < 4
    const BaseField t14 = X2 + Z2;                                               // t14 ← X2 + Z2     < 4
    const BaseField t15 = t13 * t14;                                             // t15 ← t13 · t14   < 3
    const BaseField t16 = t00 + t02;                                             // t16 ← t00 + t02   < 4
    const BaseField t17 = t15 - t16;                                             // t17 ← t15 − t16   < 2
    const BaseField t18 = t00 + t00;                                             // t18 ← t00 + t00   < 2
    const BaseField t19 = t18 + t00;                                             // t19 ← t18 + t00   < 2
    const BaseField t20 = BaseField::template mul_weierstrass_b<Gen, true>(t02); // t20 ← b3 · t02    < 2
    const BaseField t21 = t01 + t20;                                             // t21 ← t01 + t20   < 2
    const BaseField t22 = t01 - t20;                                             // t22 ← t01 − t20   < 2
    const BaseField t23 = BaseField::template mul_weierstrass_b<Gen, true>(t17); // t23 ← b3 · t17    < 2
    const auto t24 = t12.mul_wide(t23);                                          // t24 ← t12 · t23   < 2
    const auto t25 = t07.mul_wide(t22);                                          // t25 ← t07 · t22   < 2
    const BaseField X3 = (t25 - t24).reduce();                                   // X3 ← t25 − t24    < 2
    const auto t27 = t23.mul_wide(t19);                                          // t27 ← t23 · t19   < 2
    const auto t28 = t22.mul_wide(t21);                                          // t28 ← t22 · t21   < 2
    const BaseField Y3 = (t28 + t27).reduce();                                   // Y3 ← t28 + t27    < 2
    const auto t30 = t19.mul_wide(t07);                                          // t30 ← t19 · t07   < 2
    const auto t31 = t21.mul_wide(t12);                                          // t31 ← t21 · t12   < 2
    const BaseField Z3 = (t31 + t30).reduce();                                   // Z3 ← t31 + t30    < 2
    return {X3, Y3, Z3};
  }

  HOST_DEVICE_INLINE Projective operator-(const Projective& p2) const { return *this + p2.neg(); }

  HOST_DEVICE Projective operator+(const Affine<BaseField>& p2) const
  {
    const BaseField X1 = x;                                                      //                   < 2
    const BaseField Y1 = y;                                                      //                   < 2
    const BaseField Z1 = z;                                                      //                   < 2
    const BaseField X2 = p2.x;                                                   //                   < 2
    const BaseField Y2 = p2.y;                                                   //                   < 2
    const BaseField t00 = X1 * X2;                                               // t00 ← X1 · X2     < 2
    const BaseField t01 = Y1 * Y2;                                               // t01 ← Y1 · Y2     < 2
    const BaseField t02 = Z1;                                                    // t02 ← Z1          < 2
    const BaseField t03 = X1 + Y1;                                               // t03 ← X1 + Y1     < 4
    const BaseField t04 = X2 + Y2;                                               // t04 ← X2 + Y2     < 4
    const BaseField t05 = t03 * t04;                                             // t03 ← t03 · t04   < 3
    const BaseField t06 = t00 + t01;                                             // t06 ← t00 + t01   < 4
    const BaseField t07 = t05 - t06;                                             // t05 ← t05 − t06   < 2
    const BaseField t08 = Y1 + Z1;                                               // t08 ← Y1 + Z1     < 4
    const BaseField t09 = Y2 + BaseField::one();                                 // t09 ← Y2 + 1      < 4
    const BaseField t10 = t08 * t09;                                             // t10 ← t08 · t09   < 3
    const BaseField t11 = t01 + t02;                                             // t11 ← t01 + t02   < 4
    const BaseField t12 = t10 - t11;                                             // t12 ← t10 − t11   < 2
    const BaseField t13 = X1 + Z1;                                               // t13 ← X1 + Z1     < 4
    const BaseField t14 = X2 + BaseField::one();                                 // t14 ← X2 + 1      < 4
    const BaseField t15 = t13 * t14;                                             // t15 ← t13 · t14   < 3
    const BaseField t16 = t00 + t02;                                             // t16 ← t00 + t02   < 4
    const BaseField t17 = t15 - t16;                                             // t17 ← t15 − t16   < 2
    const BaseField t18 = t00 + t00;                                             // t18 ← t00 + t00   < 2
    const BaseField t19 = t18 + t00;                                             // t19 ← t18 + t00   < 2
    const BaseField t20 = BaseField::template mul_weierstrass_b<Gen, true>(t02); // t20 ← b3 · t02    < 2
    const BaseField t21 = t01 + t20;                                             // t21 ← t01 + t20   < 2
    const BaseField t22 = t01 - t20;                                             // t22 ← t01 − t20   < 2
    const BaseField t23 = BaseField::template mul_weierstrass_b<Gen, true>(t17); // t23 ← b3 · t17    < 2
    const auto t24 = t12.mul_wide(t23);                                          // t24 ← t12 · t23   < 2
    const auto t25 = t07.mul_wide(t22);                                          // t25 ← t07 · t22   < 2
    const BaseField X3 = (t25 - t24).reduce();                                   // X3 ← t25 − t24    < 2
    const auto t27 = t23.mul_wide(t19);                                          // t27 ← t23 · t19   < 2
    const auto t28 = t22.mul_wide(t21);                                          // t28 ← t22 · t21   < 2
    const BaseField Y3 = (t28 + t27).reduce();                                   // Y3 ← t28 + t27    < 2
    const auto t30 = t19.mul_wide(t07);                                          // t30 ← t19 · t07   < 2
    const auto t31 = t21.mul_wide(t12);                                          // t31 ← t21 · t12   < 2
    const BaseField Z3 = (t31 + t30).reduce();                                   // Z3 ← t31 + t30    < 2
    return {X3, Y3, Z3};
  }

  HOST_DEVICE_INLINE Projective operator-(const Affine<BaseField>& p2) const { return *this + p2.neg(); }

  HOST_DEVICE_INLINE Projective operator*(ScalarField scalar) const
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

    constexpr int nof_windows = (ScalarField::NBITS + window_size - 1) / window_size;
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

  friend HOST_DEVICE Projective operator*(ScalarField scalar, const Projective& point) { return point * scalar; }

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

  HOST_DEVICE_INLINE bool is_zero() const
  {
    return x == BaseField::zero() && y != BaseField::zero() && z == BaseField::zero();
  }

  HOST_DEVICE_INLINE bool is_on_curve() const
  {
    if (is_zero()) return true;
    bool eq_holds = (BaseField::template mul_weierstrass_b<Gen>(z.sqr() * z) + x.sqr() * x == z * y.sqr());
    return z != BaseField::zero() && eq_holds;
  }
};

#ifdef __CUDACC__
template <typename BaseField, class ScalarField, typename Gen>
struct SharedMemory<Projective<BaseField, ScalarField, Gen>> {
  __device__ Projective<BaseField, ScalarField, Gen>* getPointer()
  {
    extern __shared__ Projective<BaseField, ScalarField, Gen> s_projective_[];
    return s_projective_;
  }
};
#endif // __CUDACC__