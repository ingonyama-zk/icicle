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

    friend HOST_DEVICE_INLINE Wide operator+(const Wide& xs, const Wide& ys)
    {
      return Wide{xs.c0 + ys.c0, xs.c1 + ys.c1, xs.c2 + ys.c2};
    }

    friend HOST_DEVICE_INLINE Wide operator-(const Wide& xs, const Wide& ys)
    {
      return Wide{xs.c0 - ys.c0, xs.c1 - ys.c1, xs.c2 - ys.c2};
    }

    static constexpr HOST_DEVICE_INLINE Wide neg(const Wide& xs)
    {
      return Wide{FWide::neg(xs.c0), FWide::neg(xs.c1), FWide::neg(xs.c2)};
    }
  };

  typedef T FF;
  static constexpr unsigned TLC = 3 * FF::TLC;

  FF c0;
  FF c1;
  FF c2;

  static constexpr HOST_DEVICE_INLINE CubicExtensionField zero()
  {
    return CubicExtensionField{FF::zero(), FF::zero(), FF::zero()};
  }

  static constexpr HOST_DEVICE_INLINE CubicExtensionField one()
  {
    return CubicExtensionField{FF::one(), FF::zero(), FF::zero()};
  }

  static constexpr HOST_DEVICE_INLINE CubicExtensionField from(uint32_t val)
  {
    return CubicExtensionField{FF::from(val), FF::zero(), FF::zero()};
  }

  static constexpr HOST_DEVICE_INLINE CubicExtensionField to_montgomery(const CubicExtensionField& xs)
  {
    return CubicExtensionField{FF::to_montgomery(xs.c0), FF::to_montgomery(xs.c1), FF::to_montgomery(xs.c2)};
  }

  static constexpr HOST_DEVICE_INLINE CubicExtensionField from_montgomery(const CubicExtensionField& xs)
  {
    return CubicExtensionField{FF::from_montgomery(xs.c0), FF::from_montgomery(xs.c1), FF::from_montgomery(xs.c2)};
  }

  static HOST_INLINE CubicExtensionField rand_host()
  {
    return CubicExtensionField{FF::rand_host(), FF::rand_host(), FF::rand_host()};
  }

  static void rand_host_many(CubicExtensionField* out, int size)
  {
    for (int i = 0; i < size; i++)
      out[i] = rand_host();
  }

  template <unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE CubicExtensionField sub_modulus(const CubicExtensionField& xs)
  {
    return CubicExtensionField{
      FF::sub_modulus<REDUCTION_SIZE>(&xs.c0), FF::sub_modulus<REDUCTION_SIZE>(&xs.c1),
      FF::sub_modulus<REDUCTION_SIZE>(&xs.c2)};
  }

  friend std::ostream& operator<<(std::ostream& os, const CubicExtensionField& xs)
  {
    os << "{ c0: " << xs.c0 << " }; { c1: " << xs.c1 << " }; { c2: " << xs.c2 << " }";
    return os;
  }

  friend HOST_DEVICE_INLINE CubicExtensionField operator+(CubicExtensionField xs, const CubicExtensionField& ys)
  {
    return CubicExtensionField{xs.c0 + ys.c0, xs.c1 + ys.c1, xs.c2 + ys.c2};
  }

  friend HOST_DEVICE_INLINE CubicExtensionField operator-(CubicExtensionField xs, const CubicExtensionField& ys)
  {
    return CubicExtensionField{xs.c0 - ys.c0, xs.c1 - ys.c1, xs.c2 - ys.c2};
  }

  friend HOST_DEVICE_INLINE CubicExtensionField operator+(FF xs, const CubicExtensionField& ys)
  {
    return CubicExtensionField{xs + ys.c0, ys.c1, ys.c2};
  }

  friend HOST_DEVICE_INLINE CubicExtensionField operator-(FF xs, const CubicExtensionField& ys)
  {
    return CubicExtensionField{xs - ys.c0, FF::neg(ys.c1), FF::neg(ys.c2)};
  }

  friend HOST_DEVICE_INLINE CubicExtensionField operator+(CubicExtensionField xs, const FF& ys)
  {
    return CubicExtensionField{xs.c0 + ys, xs.c1, xs.c2};
  }

  friend HOST_DEVICE_INLINE CubicExtensionField operator-(CubicExtensionField xs, const FF& ys)
  {
    return CubicExtensionField{xs.c0 - ys, xs.c1, xs.c2};
  }

  constexpr HOST_DEVICE_INLINE CubicExtensionField operator-() const
  {
    return CubicExtensionField{FF::neg(c0), FF::neg(c1), FF::neg(c2)};
  }

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

  constexpr HOST_DEVICE_INLINE CubicExtensionField& operator+=(const FF& ys)
  {
    *this = *this + ys;
    return *this;
  }

  constexpr HOST_DEVICE_INLINE CubicExtensionField& operator-=(const FF& ys)
  {
    *this = *this - ys;
    return *this;
  }

  constexpr HOST_DEVICE_INLINE CubicExtensionField& operator*=(const FF& ys)
  {
    *this = *this * ys;
    return *this;
  }

  static constexpr HOST_DEVICE FF mul_by_nonresidue(const FF& xs)
  {
    if constexpr (CONFIG::nonresidue_is_u32) {
      return FF::template mul_unsigned<CONFIG::nonresidue>(xs);
    } else {
      return FF::template mul_const<CONFIG::nonresidue>(xs);
    }
  }

  static constexpr HOST_DEVICE FWide mul_by_nonresidue(const FWide& xs)
  {
    if constexpr (CONFIG::nonresidue_is_u32) {
      return FF::template mul_unsigned<CONFIG::nonresidue>(xs);
    } else {
      return FF::template mul_wide<>(FF::reduce(xs), CONFIG::nonresidue);
    }
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE Wide mul_wide(const CubicExtensionField& xs, const CubicExtensionField& ys)
  {
    FWide c0_prod = FF::mul_wide(xs.c0, ys.c0);
    FWide c1_prod = FF::mul_wide(xs.c1, ys.c1);
    FWide c2_prod = FF::mul_wide(xs.c2, ys.c2);

    FWide prod_of_low_sums = FF::mul_wide(xs.c0 + xs.c1, ys.c0 + ys.c1);
    FWide prod_of_high_sums = FF::mul_wide(xs.c1 + xs.c2, ys.c1 + ys.c2);
    FWide prod_of_cross_sums = FF::mul_wide(xs.c0 + xs.c2, ys.c0 + ys.c2);

    FWide nonresidue_times_a_coeff = mul_by_nonresidue(prod_of_high_sums - c1_prod - c2_prod);
    FWide nonresidue_times_c2_prod = mul_by_nonresidue(c2_prod);

    nonresidue_times_a_coeff =
      CONFIG::nonresidue_is_negative ? FWide::neg(nonresidue_times_a_coeff) : nonresidue_times_a_coeff;
    nonresidue_times_c2_prod =
      CONFIG::nonresidue_is_negative ? FWide::neg(nonresidue_times_c2_prod) : nonresidue_times_c2_prod;

    return Wide{
      c0_prod + nonresidue_times_a_coeff, prod_of_low_sums - c0_prod - c1_prod + nonresidue_times_c2_prod,
      prod_of_cross_sums - c0_prod - c2_prod + c1_prod};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE Wide mul_wide(const CubicExtensionField& xs, const FF& ys)
  {
    return Wide{FF::mul_wide(xs.c0, ys), FF::mul_wide(xs.c1, ys), FF::mul_wide(xs.c2, ys)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE Wide mul_wide(const FF& xs, const CubicExtensionField& ys)
  {
    return mul_wide(ys, xs);
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE CubicExtensionField reduce(const Wide& xs)
  {
    return CubicExtensionField{
      FF::template reduce<MODULUS_MULTIPLE>(xs.c0), FF::template reduce<MODULUS_MULTIPLE>(xs.c1),
      FF::template reduce<MODULUS_MULTIPLE>(xs.c2)};
  }

  template <class T2>
  friend HOST_DEVICE_INLINE CubicExtensionField operator*(const CubicExtensionField& xs, const T2& ys)
  {
    Wide xy = mul_wide(xs, ys);
    return reduce(xy);
  }

  friend HOST_DEVICE_INLINE bool operator==(const CubicExtensionField& xs, const CubicExtensionField& ys)
  {
    return (xs.c0 == ys.c0) && (xs.c1 == ys.c1) && (xs.c2 == ys.c2);
  }

  friend HOST_DEVICE_INLINE bool operator!=(const CubicExtensionField& xs, const CubicExtensionField& ys)
  {
    return !(xs == ys);
  }

  template <const CubicExtensionField& multiplier>
  static HOST_DEVICE CubicExtensionField mul_const(const CubicExtensionField& xs)
  {
    static constexpr FF mul_c0 = multiplier.c0;
    static constexpr FF mul_c1 = multiplier.c1;
    static constexpr FF mul_c2 = multiplier.c2;

    FWide c0_prod = FWide::from_field(FF::template mul_const<mul_c0>(xs.c0));
    FWide c1_prod = FWide::from_field(FF::template mul_const<mul_c1>(xs.c1));
    FWide c2_prod = FWide::from_field(FF::template mul_const<mul_c2>(xs.c2));

    FWide prod_of_low_sums = FF::mul_wide(xs.c0 + xs.c1, mul_c0 + mul_c1);
    FWide prod_of_high_sums = FF::mul_wide(xs.c1 + xs.c2, mul_c1 + mul_c2);
    FWide prod_of_cross_sums = FF::mul_wide(xs.c0 + xs.c2, mul_c0 + mul_c2);

    // TODO: optimize multiplication by nonresidue
    FWide nonresidue_times_a_coeff = mul_by_nonresidue(prod_of_high_sums - c1_prod - c2_prod);
    FWide nonresidue_times_c2_prod = mul_by_nonresidue(c2_prod);

    nonresidue_times_a_coeff =
      CONFIG::nonresidue_is_negative ? FWide::neg(nonresidue_times_a_coeff) : nonresidue_times_a_coeff;
    nonresidue_times_c2_prod =
      CONFIG::nonresidue_is_negative ? FWide::neg(nonresidue_times_c2_prod) : nonresidue_times_c2_prod;

    return CubicExtensionField::reduce(Wide{
      c0_prod + nonresidue_times_a_coeff, prod_of_low_sums - c0_prod - c1_prod + nonresidue_times_c2_prod,
      prod_of_cross_sums - c0_prod - c2_prod + c1_prod});
  }

  template <uint32_t multiplier, unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE CubicExtensionField mul_unsigned(const CubicExtensionField& xs)
  {
    return {
      FF::template mul_unsigned<multiplier>(xs.c0), FF::template mul_unsigned<multiplier>(xs.c1),
      FF::template mul_unsigned<multiplier>(xs.c2)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Wide sqr_wide(const CubicExtensionField& xs)
  {
    // TODO: change to a more efficient squaring
    return mul_wide<MODULUS_MULTIPLE>(xs, xs);
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE CubicExtensionField sqr(const CubicExtensionField& xs)
  {
    // TODO: change to a more efficient squaring
    return xs * xs;
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE CubicExtensionField neg(const CubicExtensionField& xs)
  {
    return CubicExtensionField{FF::neg(xs.c0), FF::neg(xs.c1), FF::neg(xs.c2)};
  }

  static constexpr HOST_DEVICE CubicExtensionField inverse(const CubicExtensionField& xs)
  {
    const FF t0 = FF::sqr(xs.c0);
    const FF t1 = FF::sqr(xs.c1);
    const FF t2 = FF::sqr(xs.c2);
    const FF t3 = xs.c0 * xs.c1;
    const FF t4 = xs.c0 * xs.c2;
    const FF t5 = xs.c1 * xs.c2;
    const FF n5 = mul_by_nonresidue(t5);

    const FF s0 = t0 - n5;
    const FF s1 = mul_by_nonresidue(t2) - t3;
    const FF s2 = t1 - t4;
    const FF a1 = xs.c2 * s1;
    const FF a2 = xs.c1 * s2;
    FF a3 = mul_by_nonresidue(a1 + a2);
    const FF t6 = FF::inverse(xs.c0 * s0 + a3);

    return CubicExtensionField{
      t6 * s0,
      t6 * s1,
      t6 * s2,
    };
  }

  static constexpr HOST_DEVICE CubicExtensionField pow(CubicExtensionField base, int exp)
  {
    CubicExtensionField res = one();
    while (exp > 0) {
      if (exp & 1) res = res * base;
      base = base * base;
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