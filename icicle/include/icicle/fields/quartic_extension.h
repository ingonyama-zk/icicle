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

    static constexpr Wide HOST_DEVICE_INLINE from_field(const QuarticExtensionField& xs)
    {
      return Wide{
        FWide::from_field(xs.c0), FWide::from_field(xs.c1), FWide::from_field(xs.c2), FWide::from_field(xs.c3)};
    }

    HOST_DEVICE_INLINE Wide operator+(const Wide& ys) const
    {
      return Wide{c0 + ys.c0, c1 + ys.c1, c2 + ys.c2, c3 + ys.c3};
    }

    HOST_DEVICE_INLINE Wide operator-(const Wide& ys) const
    {
      return Wide{c0 - ys.c0, c1 - ys.c1, c2 - ys.c2, c3 - ys.c3};
    }

    constexpr HOST_DEVICE_INLINE Wide neg() const { return Wide{c0.neg(), c1.neg(), c2.neg(), c3.neg()}; }

    constexpr HOST_DEVICE_INLINE QuarticExtensionField reduce() const
    {
      return QuarticExtensionField{c0.reduce(), c1.reduce(), c2.reduce(), c3.reduce()};
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

  constexpr HOST_DEVICE_INLINE QuarticExtensionField to_montgomery() const
  {
    return QuarticExtensionField{c0.to_montgomery(), c1.to_montgomery(), c2.to_montgomery(), c3.to_montgomery()};
  }

  constexpr HOST_DEVICE_INLINE QuarticExtensionField from_montgomery() const
  {
    return QuarticExtensionField{
      c0.from_montgomery(), c1.from_montgomery(), c2.from_montgomery(), c3.from_montgomery()};
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
  constexpr HOST_DEVICE_INLINE QuarticExtensionField sub_modulus() const
  {
    return QuarticExtensionField{
      c0.template sub_modulus<REDUCTION_SIZE>(), c1.template sub_modulus<REDUCTION_SIZE>(),
      c2.template sub_modulus<REDUCTION_SIZE>(), c3.template sub_modulus<REDUCTION_SIZE>()};
  }

  friend std::ostream& operator<<(std::ostream& os, const QuarticExtensionField& xs)
  {
    os << "{ Real: " << xs.c0 << " }; { Im1: " << xs.c1 << " }; { Im2: " << xs.c2 << " }; { Im3: " << xs.c3 << " };";
    return os;
  }

  constexpr HOST_DEVICE_INLINE QuarticExtensionField operator+(const QuarticExtensionField& ys) const
  {
    return QuarticExtensionField{c0 + ys.c0, c1 + ys.c1, c2 + ys.c2, c3 + ys.c3};
  }

  constexpr HOST_DEVICE_INLINE QuarticExtensionField operator-(const QuarticExtensionField& ys) const
  {
    return QuarticExtensionField{c0 - ys.c0, c1 - ys.c1, c2 - ys.c2, c3 - ys.c3};
  }

  constexpr HOST_DEVICE_INLINE QuarticExtensionField operator+(const FF& ys) const
  {
    return QuarticExtensionField{c0 + ys, c1, c2, c3};
  }

  constexpr HOST_DEVICE_INLINE QuarticExtensionField operator-(const FF& ys) const
  {
    return QuarticExtensionField{c0 - ys, c1, c2, c3};
  }

  constexpr HOST_DEVICE_INLINE QuarticExtensionField operator-() const { return neg(); }

  constexpr HOST_DEVICE_INLINE QuarticExtensionField& operator+=(const QuarticExtensionField& ys)
  {
    *this = *this + ys;
    return *this;
  }

  constexpr HOST_DEVICE_INLINE QuarticExtensionField& operator-=(const QuarticExtensionField& ys)
  {
    *this = *this - ys;
    return *this;
  }

  constexpr HOST_DEVICE_INLINE QuarticExtensionField& operator*=(const QuarticExtensionField& ys)
  {
    *this = *this * ys;
    return *this;
  }

  constexpr HOST_DEVICE_INLINE QuarticExtensionField& operator+=(const FF& ys)
  {
    *this = *this + ys;
    return *this;
  }

  constexpr HOST_DEVICE_INLINE QuarticExtensionField& operator-=(const FF& ys)
  {
    *this = *this - ys;
    return *this;
  }

  constexpr HOST_DEVICE_INLINE QuarticExtensionField& operator*=(const FF& ys)
  {
    *this = *this * ys;
    return *this;
  }

  constexpr HOST_DEVICE_INLINE Wide mul_wide(const QuarticExtensionField& ys) const
  {
    if (CONFIG::nonresidue_is_negative)
      return Wide{
        c0.mul_wide(ys.c0) -
          FF::template mul_unsigned<CONFIG::nonresidue>(c1.mul_wide(ys.c3) + c2.mul_wide(ys.c2) + c3.mul_wide(ys.c1)),
        c0.mul_wide(ys.c1) + c1.mul_wide(ys.c0) -
          FF::template mul_unsigned<CONFIG::nonresidue>(c2.mul_wide(ys.c3) + c3.mul_wide(ys.c2)),
        c0.mul_wide(ys.c2) + c1.mul_wide(ys.c1) + c2.mul_wide(ys.c0) -
          FF::template mul_unsigned<CONFIG::nonresidue>(c3.mul_wide(ys.c3)),
        c0.mul_wide(ys.c3) + c1.mul_wide(ys.c2) + c2.mul_wide(ys.c1) + c3.mul_wide(ys.c0)};
    else
      return Wide{
        c0.mul_wide(ys.c0) +
          FF::template mul_unsigned<CONFIG::nonresidue>(c1.mul_wide(ys.c3) + c2.mul_wide(ys.c2) + c3.mul_wide(ys.c1)),
        c0.mul_wide(ys.c1) + c1.mul_wide(ys.c0) +
          FF::template mul_unsigned<CONFIG::nonresidue>(c2.mul_wide(ys.c3) + c3.mul_wide(ys.c2)),
        c0.mul_wide(ys.c2) + c1.mul_wide(ys.c1) + c2.mul_wide(ys.c0) +
          FF::template mul_unsigned<CONFIG::nonresidue>(c3.mul_wide(ys.c3)),
        c0.mul_wide(ys.c3) + c1.mul_wide(ys.c2) + c2.mul_wide(ys.c1) + c3.mul_wide(ys.c0)};
  }

  constexpr HOST_DEVICE_INLINE Wide mul_wide(const FF& ys) const
  {
    return Wide{c0.mul_wide(ys), c1.mul_wide(ys), c2.mul_wide(ys), c3.mul_wide(ys)};
  }

  constexpr HOST_DEVICE_INLINE QuarticExtensionField operator*(const QuarticExtensionField& ys) const
  {
    Wide xy = mul_wide(ys);
    return xy.reduce();
  }

  constexpr HOST_DEVICE_INLINE QuarticExtensionField operator*(const FF& ys) const
  {
    Wide xy = mul_wide(ys);
    return xy.reduce();
  }

  friend HOST_DEVICE_INLINE QuarticExtensionField operator*(const FF& xs, const QuarticExtensionField& ys)
  {
    return ys * xs;
  }

  constexpr HOST_DEVICE_INLINE bool operator==(const QuarticExtensionField& ys) const
  {
    return (c0 == ys.c0) && (c1 == ys.c1) && (c2 == ys.c2) && (c3 == ys.c3);
  }

  constexpr HOST_DEVICE_INLINE bool operator!=(const QuarticExtensionField& ys) const { return !(*this == ys); }

  template <uint32_t multiplier, unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE QuarticExtensionField mul_unsigned(const QuarticExtensionField& xs)
  {
    return {
      FF::template mul_unsigned<multiplier>(xs.c0), FF::template mul_unsigned<multiplier>(xs.c1),
      FF::template mul_unsigned<multiplier>(xs.c2), FF::template mul_unsigned<multiplier>(xs.c3)};
  }

  constexpr HOST_DEVICE_INLINE Wide sqr_wide() const
  {
    // TODO: change to a more efficient squaring
    return mul_wide(*this);
  }

  constexpr HOST_DEVICE_INLINE QuarticExtensionField sqr() const
  {
    // TODO: change to a more efficient squaring
    return *this * *this;
  }

  constexpr HOST_DEVICE_INLINE QuarticExtensionField neg() const { return {c0.neg(), c1.neg(), c2.neg(), c3.neg()}; }

  // inverse of zero is set to be zero which is what we want most of the time
  constexpr HOST_DEVICE_INLINE QuarticExtensionField inverse() const
  {
    FF x, x0, x2;
    if (CONFIG::nonresidue_is_negative) {
      x0 =
        FF::reduce(c0.sqr_wide() + FF::template mul_unsigned<CONFIG::nonresidue>(c1.mul_wide(c3 + c3) - c2.sqr_wide()));
      x2 =
        FF::reduce(c0.mul_wide(c2 + c2) - c1.sqr_wide() + FF::template mul_unsigned<CONFIG::nonresidue>(c3.sqr_wide()));
      x = FF::reduce(x0.sqr_wide() + FF::template mul_unsigned<CONFIG::nonresidue>(x2.sqr_wide()));
    } else {
      x0 =
        FF::reduce(c0.sqr_wide() - FF::template mul_unsigned<CONFIG::nonresidue>(c1.mul_wide(c3 + c3) - c2.sqr_wide()));
      x2 =
        FF::reduce(c0.mul_wide(c2 + c2) - c1.sqr_wide() - FF::template mul_unsigned<CONFIG::nonresidue>(c3.sqr_wide()));
      x = FF::reduce(x0.sqr_wide() - FF::template mul_unsigned<CONFIG::nonresidue>(x2.sqr_wide()));
    }
    FF x_inv = x.inverse();
    x0 = x0 * x_inv;
    x2 = x2 * x_inv;
    return {
      FF::reduce(
        (CONFIG::nonresidue_is_negative
           ? (c0.mul_wide(x0) + FF::template mul_unsigned<CONFIG::nonresidue>(c2.mul_wide(x2)))
           : (c0.mul_wide(x0)) - FF::template mul_unsigned<CONFIG::nonresidue>(c2.mul_wide(x2)))),
      FF::reduce(
        (CONFIG::nonresidue_is_negative ? FF::template mul_unsigned<CONFIG::nonresidue>(c3.mul_wide(x2)).neg()
                                        : FF::template mul_unsigned<CONFIG::nonresidue>(c3.mul_wide(x2))) -
        c1.mul_wide(x0)),
      FF::reduce(c2.mul_wide(x0) - c0.mul_wide(x2)),
      FF::reduce(c1.mul_wide(x2) - c3.mul_wide(x0)),
    };
  }

  constexpr HOST_DEVICE QuarticExtensionField pow(int exp) const
  {
    QuarticExtensionField res = one();
    QuarticExtensionField base = *this;
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
