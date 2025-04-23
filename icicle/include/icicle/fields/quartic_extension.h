#pragma once

#include "icicle/errors.h"
#include "icicle/fields/field.h"
#include "icicle/utils/modifiers.h"

#if __CUDACC__
  #include "gpu-utils/sharedmem.h"
#endif // __CUDACC__

template <typename CONFIG, class T>
class QuarticExtensionField
{
private:
  typedef typename T::Wide FWide;

public:
  struct Wide {
    FWide c0;
    FWide c1;
    FWide c2;
    FWide c3;

    friend HOST_DEVICE_INLINE Wide operator+(Wide xs, const Wide& ys)
    {
      return Wide{xs.c0 + ys.c0, xs.c1 + ys.c1, xs.c2 + ys.c2, xs.c3 + ys.c3};
    }

    friend HOST_DEVICE_INLINE Wide operator-(Wide xs, const Wide& ys)
    {
      return Wide{xs.c0 - ys.c0, xs.c1 - ys.c1, xs.c2 - ys.c2, xs.c3 - ys.c3};
    }

    static constexpr HOST_DEVICE_INLINE Wide neg(const Wide& xs)
    {
      return Wide{FWide::neg(xs.c0), FWide::neg(xs.c1), FWide::neg(xs.c2), FWide::neg(xs.c3)};
    }
  };

  typedef T FF;
  static constexpr unsigned TLC = 4 * FF::TLC;

  FF c0;
  FF c1;
  FF c2;
  FF c3;

  static constexpr HOST_DEVICE_INLINE QuarticExtensionField zero()
  {
    return QuarticExtensionField{FF::zero(), FF::zero(), FF::zero(), FF::zero()};
  }

  static constexpr HOST_DEVICE_INLINE QuarticExtensionField one()
  {
    return QuarticExtensionField{FF::one(), FF::zero(), FF::zero(), FF::zero()};
  }

  // Converts a uint32_t value to a QuarticExtensionField element.
  // If `val` ≥ p, it wraps around modulo p, affecting only the first coefficient.
  static constexpr HOST_DEVICE_INLINE QuarticExtensionField from(uint32_t val)
  {
    return QuarticExtensionField{FF::from(val), FF::zero(), FF::zero(), FF::zero()};
  }

  static constexpr HOST_DEVICE_INLINE QuarticExtensionField to_montgomery(const QuarticExtensionField& xs)
  {
    return QuarticExtensionField{
      FF::to_montgomery(xs.c0), FF::to_montgomery(xs.c1), FF::to_montgomery(xs.c2), FF::to_montgomery(xs.c3)};
  }

  static constexpr HOST_DEVICE_INLINE QuarticExtensionField from_montgomery(const QuarticExtensionField& xs)
  {
    return QuarticExtensionField{
      FF::from_montgomery(xs.c0), FF::from_montgomery(xs.c1), FF::from_montgomery(xs.c2), FF::from_montgomery(xs.c3)};
  }

  static HOST_INLINE QuarticExtensionField rand_host()
  {
    return QuarticExtensionField{FF::rand_host(), FF::rand_host(), FF::rand_host(), FF::rand_host()};
  }

  static void rand_host_many(QuarticExtensionField* out, int size)
  {
    for (int i = 0; i < size; i++)
      out[i] = rand_host();
  }

  template <unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE QuarticExtensionField sub_modulus(const QuarticExtensionField& xs)
  {
    return QuarticExtensionField{
      FF::sub_modulus<REDUCTION_SIZE>(&xs.c0), FF::sub_modulus<REDUCTION_SIZE>(&xs.c1),
      FF::sub_modulus<REDUCTION_SIZE>(&xs.c2), FF::sub_modulus<REDUCTION_SIZE>(&xs.c3)};
  }

  friend std::ostream& operator<<(std::ostream& os, const QuarticExtensionField& xs)
  {
    os << "{ Real: " << xs.c0 << " }; { Im1: " << xs.c1 << " }; { Im2: " << xs.c2 << " }; { Im3: " << xs.c3 << " };";
    return os;
  }

  friend HOST_DEVICE_INLINE QuarticExtensionField operator+(QuarticExtensionField xs, const QuarticExtensionField& ys)
  {
    return QuarticExtensionField{xs.c0 + ys.c0, xs.c1 + ys.c1, xs.c2 + ys.c2, xs.c3 + ys.c3};
  }

  friend HOST_DEVICE_INLINE QuarticExtensionField operator-(QuarticExtensionField xs, const QuarticExtensionField& ys)
  {
    return QuarticExtensionField{xs.c0 - ys.c0, xs.c1 - ys.c1, xs.c2 - ys.c2, xs.c3 - ys.c3};
  }

  friend HOST_DEVICE_INLINE QuarticExtensionField operator+(FF xs, const QuarticExtensionField& ys)
  {
    return QuarticExtensionField{xs + ys.c0, ys.c1, ys.c2, ys.c3};
  }

  friend HOST_DEVICE_INLINE QuarticExtensionField operator-(FF xs, const QuarticExtensionField& ys)
  {
    return QuarticExtensionField{xs - ys.c0, FF::neg(ys.c1), FF::neg(ys.c2), FF::neg(ys.c3)};
  }

  friend HOST_DEVICE_INLINE QuarticExtensionField operator+(QuarticExtensionField xs, const FF& ys)
  {
    return QuarticExtensionField{xs.c0 + ys, xs.c1, xs.c2, xs.c3};
  }

  friend HOST_DEVICE_INLINE QuarticExtensionField operator-(QuarticExtensionField xs, const FF& ys)
  {
    return QuarticExtensionField{xs.c0 - ys, xs.c1, xs.c2, xs.c3};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Wide mul_wide(const QuarticExtensionField& xs, const QuarticExtensionField& ys)
  {
    if (CONFIG::nonresidue_is_negative)
      return Wide{
        FF::mul_wide(xs.c0, ys.c0) -
          FF::template mul_unsigned<CONFIG::nonresidue>(
            FF::mul_wide(xs.c1, ys.c3) + FF::mul_wide(xs.c2, ys.c2) + FF::mul_wide(xs.c3, ys.c1)),
        FF::mul_wide(xs.c0, ys.c1) + FF::mul_wide(xs.c1, ys.c0) -
          FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.c2, ys.c3) + FF::mul_wide(xs.c3, ys.c2)),
        FF::mul_wide(xs.c0, ys.c2) + FF::mul_wide(xs.c1, ys.c1) + FF::mul_wide(xs.c2, ys.c0) -
          FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.c3, ys.c3)),
        FF::mul_wide(xs.c0, ys.c3) + FF::mul_wide(xs.c1, ys.c2) + FF::mul_wide(xs.c2, ys.c1) +
          FF::mul_wide(xs.c3, ys.c0)};
    else
      return Wide{
        FF::mul_wide(xs.c0, ys.c0) +
          FF::template mul_unsigned<CONFIG::nonresidue>(
            FF::mul_wide(xs.c1, ys.c3) + FF::mul_wide(xs.c2, ys.c2) + FF::mul_wide(xs.c3, ys.c1)),
        FF::mul_wide(xs.c0, ys.c1) + FF::mul_wide(xs.c1, ys.c0) +
          FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.c2, ys.c3) + FF::mul_wide(xs.c3, ys.c2)),
        FF::mul_wide(xs.c0, ys.c2) + FF::mul_wide(xs.c1, ys.c1) + FF::mul_wide(xs.c2, ys.c0) +
          FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.c3, ys.c3)),
        FF::mul_wide(xs.c0, ys.c3) + FF::mul_wide(xs.c1, ys.c2) + FF::mul_wide(xs.c2, ys.c1) +
          FF::mul_wide(xs.c3, ys.c0)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Wide mul_wide(const QuarticExtensionField& xs, const FF& ys)
  {
    return Wide{FF::mul_wide(xs.c0, ys), FF::mul_wide(xs.c1, ys), FF::mul_wide(xs.c2, ys), FF::mul_wide(xs.c3, ys)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Wide mul_wide(const FF& xs, const QuarticExtensionField& ys)
  {
    return Wide{FF::mul_wide(xs, ys.c0), FF::mul_wide(xs, ys.c1), FF::mul_wide(xs, ys.c2), FF::mul_wide(xs, ys.c3)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE QuarticExtensionField reduce(const Wide& xs)
  {
    return QuarticExtensionField{
      FF::template reduce<MODULUS_MULTIPLE>(xs.c0), FF::template reduce<MODULUS_MULTIPLE>(xs.c1),
      FF::template reduce<MODULUS_MULTIPLE>(xs.c2), FF::template reduce<MODULUS_MULTIPLE>(xs.c3)};
  }

  template <class T1, class T2>
  friend HOST_DEVICE_INLINE QuarticExtensionField operator*(const T1& xs, const T2& ys)
  {
    Wide xy = mul_wide(xs, ys);
    return reduce(xy);
  }

  friend HOST_DEVICE_INLINE bool operator==(const QuarticExtensionField& xs, const QuarticExtensionField& ys)
  {
    return (xs.c0 == ys.c0) && (xs.c1 == ys.c1) && (xs.c2 == ys.c2) && (xs.c3 == ys.c3);
  }

  friend HOST_DEVICE_INLINE bool operator!=(const QuarticExtensionField& xs, const QuarticExtensionField& ys)
  {
    return !(xs == ys);
  }

  template <uint32_t multiplier, unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE QuarticExtensionField mul_unsigned(const QuarticExtensionField& xs)
  {
    return {
      FF::template mul_unsigned<multiplier>(xs.c0), FF::template mul_unsigned<multiplier>(xs.c1),
      FF::template mul_unsigned<multiplier>(xs.c2), FF::template mul_unsigned<multiplier>(xs.c3)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Wide sqr_wide(const QuarticExtensionField& xs)
  {
    // TODO: change to a more efficient squaring
    return mul_wide<MODULUS_MULTIPLE>(xs, xs);
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE QuarticExtensionField sqr(const QuarticExtensionField& xs)
  {
    // TODO: change to a more efficient squaring
    return xs * xs;
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE QuarticExtensionField neg(const QuarticExtensionField& xs)
  {
    return {FF::neg(xs.c0), FF::neg(xs.c1), FF::neg(xs.c2), FF::neg(xs.c3)};
  }

  // inverse of zero is set to be zero which is what we want most of the time
  static constexpr HOST_DEVICE_INLINE QuarticExtensionField inverse(const QuarticExtensionField& xs)
  {
    FF x, x0, x2;
    if (CONFIG::nonresidue_is_negative) {
      x0 = FF::reduce(
        FF::sqr_wide(xs.c0) +
        FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.c1, xs.c3 + xs.c3) - FF::sqr_wide(xs.c2)));
      x2 = FF::reduce(
        FF::mul_wide(xs.c0, xs.c2 + xs.c2) - FF::sqr_wide(xs.c1) +
        FF::template mul_unsigned<CONFIG::nonresidue>(FF::sqr_wide(xs.c3)));
      x = FF::reduce(FF::sqr_wide(x0) + FF::template mul_unsigned<CONFIG::nonresidue>(FF::sqr_wide(x2)));
    } else {
      x0 = FF::reduce(
        FF::sqr_wide(xs.c0) -
        FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.c1, xs.c3 + xs.c3) - FF::sqr_wide(xs.c2)));
      x2 = FF::reduce(
        FF::mul_wide(xs.c0, xs.c2 + xs.c2) - FF::sqr_wide(xs.c1) -
        FF::template mul_unsigned<CONFIG::nonresidue>(FF::sqr_wide(xs.c3)));
      x = FF::reduce(FF::sqr_wide(x0) - FF::template mul_unsigned<CONFIG::nonresidue>(FF::sqr_wide(x2)));
    }
    FF x_inv = FF::inverse(x);
    x0 = x0 * x_inv;
    x2 = x2 * x_inv;
    return {
      FF::reduce(
        (CONFIG::nonresidue_is_negative
           ? (FF::mul_wide(xs.c0, x0) + FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.c2, x2)))
           : (FF::mul_wide(xs.c0, x0))-FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.c2, x2)))),
      FF::reduce(
        (CONFIG::nonresidue_is_negative
           ? FWide::neg(FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.c3, x2)))
           : FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.c3, x2))) -
        FF::mul_wide(xs.c1, x0)),
      FF::reduce(FF::mul_wide(xs.c2, x0) - FF::mul_wide(xs.c0, x2)),
      FF::reduce(FF::mul_wide(xs.c1, x2) - FF::mul_wide(xs.c3, x0)),
    };
  }

  static constexpr HOST_DEVICE QuarticExtensionField pow(QuarticExtensionField base, int exp)
  {
    QuarticExtensionField res = one();
    while (exp > 0) {
      if (exp & 1) res = res * base;
      base = base * base;
      exp >>= 1;
    }
    return res;
  }

  // Receives an array of bytes and its size and returns extension field element.
  static constexpr HOST_DEVICE_INLINE QuarticExtensionField from(const std::byte* in, unsigned nof_bytes)
  {
    if (nof_bytes < 4 * sizeof(FF)) {
#ifndef __CUDACC__
      ICICLE_LOG_ERROR << "Input size is too small";
#endif // __CUDACC__
      return QuarticExtensionField::zero();
    }

    return QuarticExtensionField{
      FF::from(in, sizeof(FF)), FF::from(in + sizeof(FF), sizeof(FF)), FF::from(in + 2 * sizeof(FF), sizeof(FF)),
      FF::from(in + 3 * sizeof(FF), sizeof(FF))};
  }
};

#if __CUDACC__
template <class CONFIG, class T>
struct SharedMemory<QuarticExtensionField<CONFIG, T>> {
  __device__ QuarticExtensionField<CONFIG, T>* getPointer()
  {
    extern __shared__ QuarticExtensionField<CONFIG, T> s_ext4_scalar_[];
    return s_ext4_scalar_;
  }
};
#endif // __CUDACC__
