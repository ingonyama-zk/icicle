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

  struct ExtensionWide {
    FWide c0;
    FWide c1;
    FWide c2;

    friend HOST_DEVICE_INLINE ExtensionWide operator+(ExtensionWide xs, const ExtensionWide& ys)
    {
      return ExtensionWide{xs.c0 + ys.c0, xs.c1 + ys.c1, xs.c2 + ys.c2};
    }

    friend HOST_DEVICE_INLINE ExtensionWide operator-(ExtensionWide xs, const ExtensionWide& ys)
    {
      return ExtensionWide{xs.c0 - ys.c0, xs.c1 - ys.c1, xs.c2 - ys.c2};
    }
  };

public:
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

  //   static constexpr HOST_DEVICE_INLINE CubicExtensionField to_montgomery(const CubicExtensionField& xs)
  //   {
  //     return CubicExtensionField{xs.c0 * FF{CONFIG::montgomery_r}, xs.c1 * FF{CONFIG::montgomery_r}, xs.c2 *
  //     FF{CONFIG::montgomery_r}};
  //   }

  //   static constexpr HOST_DEVICE_INLINE CubicExtensionField from_montgomery(const CubicExtensionField& xs)
  //   {
  //     return CubicExtensionField{xs.c0 * FF{CONFIG::montgomery_r_inv}, xs.c1 * FF{CONFIG::montgomery_r_inv}, xs.c2 *
  //     FF{CONFIG::montgomery_r_inv}};
  //   }

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

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionWide
  mul_wide(const CubicExtensionField& xs, const CubicExtensionField& ys)
  {
    FWide c0_prod = FF::mul_wide(xs.c0, ys.c0);
    FWide c1_prod = FF::mul_wide(xs.c1, ys.c1);
    FWide c2_prod = FF::mul_wide(xs.c2, ys.c2);

    FWide prod_of_low_sums = FF::mul_wide(xs.c0 + xs.c1, ys.c0 + ys.c1);
    FWide prod_of_high_sums = FF::mul_wide(xs.c1 + xs.c2, ys.c1 + ys.c2);
    FWide prod_of_cross_sums = FF::mul_wide(xs.c0 + xs.c2, ys.c0 + ys.c2);

    // TODO: optimize multiplication by nonresidue
    FF nonresidue(CONFIG::nonresidue_re, CONFIG::nonresidue_im);
    FWide nonresidue_times_a_coeff = nonresidue * (prod_of_high_sums - c1_prod - c2_prod);
    FWide nonresidue_times_c2_prod = nonresidue * c2_prod;

    nonresidue_times_a_coeff =
      CONFIG::nonresidue_is_negative ? FWide::neg(nonresidue_times_a_coeff) : nonresidue_times_a_coeff;
    nonresidue_times_c2_prod =
      CONFIG::nonresidue_is_negative ? FWide::neg(nonresidue_times_c2_prod) : nonresidue_times_c2_prod;

    return ExtensionWide{
      c0_prod + nonresidue_times_a_coeff, prod_of_low_sums - c0_prod - c1_prod + nonresidue_times_c2_prod,
      prod_of_cross_sums - c0_prod - c2_prod + c1_prod};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionWide mul_wide(const CubicExtensionField& xs, const FF& ys)
  {
    return ExtensionWide{FF::mul_wide(xs.c0, ys), FF::mul_wide(xs.c1, ys), FF::mul_wide(xs.c2, ys)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionWide mul_wide(const FF& xs, const CubicExtensionField& ys)
  {
    return mul_wide(ys, xs);
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE CubicExtensionField reduce(const ExtensionWide& xs)
  {
    return CubicExtensionField{
      FF::template reduce<MODULUS_MULTIPLE>(xs.c0), FF::template reduce<MODULUS_MULTIPLE>(xs.c1),
      FF::template reduce<MODULUS_MULTIPLE>(xs.c2)};
  }

  template <class T1, class T2>
  friend HOST_DEVICE_INLINE CubicExtensionField operator*(const T1& xs, const T2& ys)
  {
    ExtensionWide xy = mul_wide(xs, ys);
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
  static HOST_DEVICE_INLINE CubicExtensionField mul_const(const CubicExtensionField& xs)
  {
    static constexpr FF mul_c0 = multiplier.c0;
    static constexpr FF mul_c1 = multiplier.c1;
    static constexpr FF mul_c2 = multiplier.c2;

    FF c0_prod = FF::template mul_const<mul_c0>(xs.c0);
    FF c1_prod = FF::template mul_const<mul_c1>(xs.c1);
    FF c2_prod = FF::template mul_const<mul_c2>(xs.c2);

    FWide prod_of_low_sums = FF::mul_wide(xs.c0 + xs.c1, mul_c0 + mul_c1);
    FWide prod_of_high_sums = FF::mul_wide(xs.c1 + xs.c2, mul_c1 + mul_c2);
    FWide prod_of_cross_sums = FF::mul_wide(xs.c0 + xs.c2, mul_c0 + mul_c2);

    // TODO: optimize multiplication by nonresidue
    FF nonresidue(CONFIG::nonresidue_re, CONFIG::nonresidue_im);
    FWide nonresidue_times_a_coeff = nonresidue * (prod_of_high_sums - c1_prod - c2_prod);
    FWide nonresidue_times_c2_prod = nonresidue * c2_prod;

    nonresidue_times_a_coeff =
      CONFIG::nonresidue_is_negative ? FWide::neg(nonresidue_times_a_coeff) : nonresidue_times_a_coeff;
    nonresidue_times_c2_prod =
      CONFIG::nonresidue_is_negative ? FWide::neg(nonresidue_times_c2_prod) : nonresidue_times_c2_prod;

    return ExtensionWide{
      c0_prod + nonresidue_times_a_coeff, prod_of_low_sums - c0_prod - c1_prod + nonresidue_times_c2_prod,
      prod_of_cross_sums - c0_prod - c2_prod + c1_prod};
  }

  template <uint32_t multiplier, unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE CubicExtensionField mul_unsigned(const CubicExtensionField& xs)
  {
    return {
      FF::template mul_unsigned<multiplier>(xs.c0), FF::template mul_unsigned<multiplier>(xs.c1),
      FF::template mul_unsigned<multiplier>(xs.c2)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionWide sqr_wide(const CubicExtensionField& xs)
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