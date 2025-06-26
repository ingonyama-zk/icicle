#pragma once

#include "field.h"
#include "icicle/utils/modifiers.h"
#ifdef __CUDACC__
  #include "gpu-utils/sharedmem.h"
#endif // __CUDACC__

template <typename CONFIG, class T>
class CubicExtensionField
{
private:
  friend T;

  typedef typename T::Wide FWide;

public:
  struct Wide {
    FWide c0;
    FWide c1;
    FWide c2;

    static constexpr Wide HOST_DEVICE_INLINE from_field(const CubicExtensionField xs)
    {
      return Wide{FWide::from_field(xs.c0), FWide::from_field(xs.c1), FWide::from_field(xs.c2)};
    }

    HOST_DEVICE_INLINE Wide operator+(const Wide& ys) const { return Wide{c0 + ys.c0, c1 + ys.c1, c2 + ys.c2}; }

    HOST_DEVICE_INLINE Wide operator-(const Wide& ys) const { return Wide{c0 - ys.c0, c1 - ys.c1, c2 - ys.c2}; }

    constexpr HOST_DEVICE_INLINE Wide neg() const { return Wide{c0.neg(), c1.neg(), c2.neg()}; }

    constexpr HOST_DEVICE_INLINE CubicExtensionField reduce() const
    {
      return CubicExtensionField{c0.reduce(), c1.reduce(), c2.reduce()};
    }
  };

  typedef T BaseField;
  static constexpr unsigned TLC = 3 * BaseField::TLC;

  BaseField c0;
  BaseField c1;
  BaseField c2;

  static constexpr HOST_DEVICE_INLINE CubicExtensionField zero()
  {
    return CubicExtensionField{BaseField::zero(), BaseField::zero(), BaseField::zero()};
  }

  static constexpr HOST_DEVICE_INLINE CubicExtensionField one()
  {
    return CubicExtensionField{BaseField::one(), BaseField::zero(), BaseField::zero()};
  }

  static constexpr HOST_DEVICE_INLINE CubicExtensionField from(uint32_t val)
  {
    return CubicExtensionField{BaseField::from(val), BaseField::zero(), BaseField::zero()};
  }

  constexpr HOST_DEVICE_INLINE CubicExtensionField to_montgomery() const
  {
    return CubicExtensionField{c0.to_montgomery(), c1.to_montgomery(), c2.to_montgomery()};
  }

  constexpr HOST_DEVICE_INLINE CubicExtensionField from_montgomery() const
  {
    return CubicExtensionField{c0.from_montgomery(), c1.from_montgomery(), c2.from_montgomery()};
  }

  static HOST_INLINE CubicExtensionField rand_host()
  {
    return CubicExtensionField{BaseField::rand_host(), BaseField::rand_host(), BaseField::rand_host()};
  }

  static void rand_host_many(CubicExtensionField* out, int size)
  {
    for (int i = 0; i < size; i++)
      out[i] = rand_host();
  }

  template <unsigned REDUCTION_SIZE = 1>
  constexpr HOST_DEVICE_INLINE CubicExtensionField sub_modulus() const
  {
    return CubicExtensionField{
      c0.template sub_modulus<REDUCTION_SIZE>(), c1.template sub_modulus<REDUCTION_SIZE>(),
      c2.template sub_modulus<REDUCTION_SIZE>()};
  }

  friend std::ostream& operator<<(std::ostream& os, const CubicExtensionField& xs)
  {
    os << "{ c0: " << xs.c0 << " }; { c1: " << xs.c1 << " }; { c2: " << xs.c2 << " }";
    return os;
  }

  HOST_DEVICE_INLINE CubicExtensionField operator+(const CubicExtensionField& ys) const
  {
    return CubicExtensionField{c0 + ys.c0, c1 + ys.c1, c2 + ys.c2};
  }

  HOST_DEVICE_INLINE CubicExtensionField operator-(const CubicExtensionField& ys) const
  {
    return CubicExtensionField{c0 - ys.c0, c1 - ys.c1, c2 - ys.c2};
  }

  friend HOST_DEVICE_INLINE CubicExtensionField operator+(BaseField xs, const CubicExtensionField& ys)
  {
    return CubicExtensionField{xs + ys.c0, ys.c1, ys.c2};
  }

  friend HOST_DEVICE_INLINE CubicExtensionField operator-(BaseField xs, const CubicExtensionField& ys)
  {
    return CubicExtensionField{xs - ys.c0, ys.c1.neg(), ys.c2.neg()};
  }

  HOST_DEVICE_INLINE CubicExtensionField operator+(const BaseField& ys) const
  {
    return CubicExtensionField{c0 + ys, c1, c2};
  }

  HOST_DEVICE_INLINE CubicExtensionField operator-(const BaseField& ys) const
  {
    return CubicExtensionField{c0 - ys, c1, c2};
  }

  constexpr HOST_DEVICE_INLINE CubicExtensionField operator-() const { return this->neg(); }

  constexpr HOST_DEVICE_INLINE CubicExtensionField& operator+=(const CubicExtensionField& ys)
  {
    *this = *this + ys;
    return *this;
  }

  constexpr HOST_DEVICE_INLINE CubicExtensionField& operator-=(const CubicExtensionField& ys)
  {
    *this = *this - ys;
    return *this;
  }

  constexpr HOST_DEVICE_INLINE CubicExtensionField& operator*=(const CubicExtensionField& ys)
  {
    *this = *this * ys;
    return *this;
  }

  constexpr HOST_DEVICE_INLINE CubicExtensionField& operator+=(const BaseField& ys)
  {
    *this = *this + ys;
    return *this;
  }

  constexpr HOST_DEVICE_INLINE CubicExtensionField& operator-=(const BaseField& ys)
  {
    *this = *this - ys;
    return *this;
  }

  constexpr HOST_DEVICE_INLINE CubicExtensionField& operator*=(const BaseField& ys)
  {
    *this = *this * ys;
    return *this;
  }

  /**
   * @brief Multiplies a field element by the nonresidue of the field
   */
  static constexpr HOST_DEVICE BaseField mul_by_nonresidue(const BaseField& xs)
  {
    if constexpr (CONFIG::nonresidue_is_u32) {
      return BaseField::template mul_unsigned<CONFIG::nonresidue>(xs);
    } else {
      return BaseField::template mul_const<CONFIG::nonresidue>(xs);
    }
  }

  /**
   * @brief Multiplies a wide field element by the nonresidue of the field
   */
  static constexpr HOST_DEVICE FWide mul_by_nonresidue(const FWide& xs)
  {
    if constexpr (CONFIG::nonresidue_is_u32) {
      return BaseField::template mul_unsigned<CONFIG::nonresidue>(xs);
    } else {
      return xs.reduce().mul_wide(CONFIG::nonresidue);
    }
  }

  constexpr HOST_DEVICE_INLINE Wide mul_wide(const CubicExtensionField& ys) const
  {
    FWide c0_prod = c0.mul_wide(ys.c0);
    FWide c1_prod = c1.mul_wide(ys.c1);
    FWide c2_prod = c2.mul_wide(ys.c2);

    FWide prod_of_low_sums = (c0 + c1).mul_wide(ys.c0 + ys.c1);
    FWide prod_of_high_sums = (c1 + c2).mul_wide(ys.c1 + ys.c2);
    FWide prod_of_cross_sums = (c0 + c2).mul_wide(ys.c0 + ys.c2);

    FWide nonresidue_times_a_coeff = mul_by_nonresidue(prod_of_high_sums - c1_prod - c2_prod);
    FWide nonresidue_times_c2_prod = mul_by_nonresidue(c2_prod);

    nonresidue_times_a_coeff =
      CONFIG::nonresidue_is_negative ? nonresidue_times_a_coeff.neg() : nonresidue_times_a_coeff;
    nonresidue_times_c2_prod =
      CONFIG::nonresidue_is_negative ? nonresidue_times_c2_prod.neg() : nonresidue_times_c2_prod;

    return Wide{
      c0_prod + nonresidue_times_a_coeff, prod_of_low_sums - c0_prod - c1_prod + nonresidue_times_c2_prod,
      prod_of_cross_sums - c0_prod - c2_prod + c1_prod};
  }

  constexpr HOST_DEVICE_INLINE Wide mul_wide(const BaseField& ys) const
  {
    return Wide{c0.mul_wide(ys), c1.mul_wide(ys), c2.mul_wide(ys)};
  }

  static constexpr HOST_DEVICE_INLINE Wide mul_wide(const BaseField& xs, const CubicExtensionField& ys)
  {
    return ys.mul_wide(xs);
  }

  friend HOST_DEVICE_INLINE CubicExtensionField operator*(const CubicExtensionField& xs, const CubicExtensionField& ys)
  {
    Wide xy = xs.mul_wide(ys);
    return xy.reduce();
  }

  template <
    class T2,
    typename = typename std::enable_if<
      !std::is_same<T2, CubicExtensionField>() && !std::is_base_of<CubicExtensionField, T2>()>::type>
  friend HOST_DEVICE_INLINE CubicExtensionField operator*(const CubicExtensionField& xs, const T2& ys)
  {
    Wide xy = xs.mul_wide(ys);
    return xy.reduce();
  }

  friend HOST_DEVICE_INLINE CubicExtensionField operator*(const CubicExtensionField& xs, const BaseField& ys)
  {
    Wide xy = xs.mul_wide(ys);
    return xy.reduce();
  }

  HOST_DEVICE_INLINE bool operator==(const CubicExtensionField& ys) const
  {
    return (c0 == ys.c0) && (c1 == ys.c1) && (c2 == ys.c2);
  }

  HOST_DEVICE_INLINE bool operator!=(const CubicExtensionField& ys) const { return !(*this == ys); }

  template <const CubicExtensionField& multiplier>
  static HOST_DEVICE CubicExtensionField mul_const(const CubicExtensionField& xs)
  {
    static constexpr BaseField mul_c0 = multiplier.c0;
    static constexpr BaseField mul_c1 = multiplier.c1;
    static constexpr BaseField mul_c2 = multiplier.c2;

    FWide c0_prod = FWide::from_field(BaseField::template mul_const<mul_c0>(xs.c0));
    FWide c1_prod = FWide::from_field(BaseField::template mul_const<mul_c1>(xs.c1));
    FWide c2_prod = FWide::from_field(BaseField::template mul_const<mul_c2>(xs.c2));

    FWide prod_of_low_sums = (xs.c0 + xs.c1).mul_wide(mul_c0 + mul_c1);
    FWide prod_of_high_sums = (xs.c1 + xs.c2).mul_wide(mul_c1 + mul_c2);
    FWide prod_of_cross_sums = (xs.c0 + xs.c2).mul_wide(mul_c0 + mul_c2);

    // TODO: optimize multiplication by nonresidue
    FWide nonresidue_times_a_coeff = mul_by_nonresidue(prod_of_high_sums - c1_prod - c2_prod);
    FWide nonresidue_times_c2_prod = mul_by_nonresidue(c2_prod);

    nonresidue_times_a_coeff =
      CONFIG::nonresidue_is_negative ? nonresidue_times_a_coeff.neg() : nonresidue_times_a_coeff;
    nonresidue_times_c2_prod =
      CONFIG::nonresidue_is_negative ? nonresidue_times_c2_prod.neg() : nonresidue_times_c2_prod;

    return Wide{
      c0_prod + nonresidue_times_a_coeff, prod_of_low_sums - c0_prod - c1_prod + nonresidue_times_c2_prod,
      prod_of_cross_sums - c0_prod - c2_prod + c1_prod}
      .reduce();
  }

  template <uint32_t multiplier>
  static constexpr HOST_DEVICE CubicExtensionField mul_unsigned(const CubicExtensionField& xs)
  {
    return {
      BaseField::template mul_unsigned<multiplier>(xs.c0), BaseField::template mul_unsigned<multiplier>(xs.c1),
      BaseField::template mul_unsigned<multiplier>(xs.c2)};
  }

  constexpr HOST_DEVICE_INLINE Wide sqr_wide() const
  {
    // TODO: change to a more efficient squaring
    return mul_wide(*this, *this);
  }

  constexpr HOST_DEVICE_INLINE CubicExtensionField sqr() const
  {
    // TODO: change to a more efficient squaring
    return *this * *this;
  }

  constexpr HOST_DEVICE_INLINE CubicExtensionField neg() const
  {
    return CubicExtensionField{c0.neg(), c1.neg(), c2.neg()};
  }

  constexpr HOST_DEVICE CubicExtensionField inverse() const
  {
    const BaseField t0 = c0.sqr();
    const BaseField t1 = c1.sqr();
    const BaseField t2 = c2.sqr();
    const BaseField t3 = c0 * c1;
    const BaseField t4 = c0 * c2;
    const BaseField t5 = c1 * c2;
    const BaseField n5 = mul_by_nonresidue(t5);

    const BaseField s0 = t0 - n5;
    const BaseField s1 = mul_by_nonresidue(t2) - t3;
    const BaseField s2 = t1 - t4;
    const BaseField a1 = c2 * s1;
    const BaseField a2 = c1 * s2;
    BaseField a3 = mul_by_nonresidue(a1 + a2);
    const BaseField t6 = (c0 * s0 + a3).inverse();

    return CubicExtensionField{
      t6 * s0,
      t6 * s1,
      t6 * s2,
    };
  }

  constexpr HOST_DEVICE CubicExtensionField pow(int exp) const
  {
    CubicExtensionField res = one();
    CubicExtensionField base = *this;
    while (exp > 0) {
      if (exp & 1) res = res * base;
      base = base.sqr();
      exp >>= 1;
    }
    return res;
  }
};

#ifdef __CUDACC__
template <typename CONFIG, class T>
struct SharedMemory<CubicExtensionField<CONFIG, T>> {
  __device__ CubicExtensionField<CONFIG, T>* getPointer()
  {
    extern __shared__ CubicExtensionField<CONFIG, T> s_ext3_scalar_[];
    return s_ext3_scalar_;
  }
};
#endif //__CUDAC__