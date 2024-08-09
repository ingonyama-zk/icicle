#pragma once

#include "field.cuh"
#include "gpu-utils/modifiers.cuh"
#include "gpu-utils/sharedmem.cuh"

template <typename CONFIG, class T>
class QuartExtensionField
{
private:
  typedef typename T::Wide FWide;

  struct ExtensionWide {
    FWide real;
    FWide im1;
    FWide im2;
    FWide im3;

    friend HOST_DEVICE_INLINE ExtensionWide operator+(ExtensionWide xs, const ExtensionWide& ys)
    {
      return ExtensionWide{xs.real + ys.real, xs.im1 + ys.im1, xs.im2 + ys.im2, xs.im3 + ys.im3};
    }

    friend HOST_DEVICE_INLINE ExtensionWide operator-(ExtensionWide xs, const ExtensionWide& ys)
    {
      return ExtensionWide{xs.real - ys.real, xs.im1 - ys.im1, xs.im2 - ys.im2, xs.im3 - ys.im3};
    }
  };

public:
  typedef T FF;
  static constexpr unsigned TLC = 4 * CONFIG::limbs_count;

  FF real;
  FF im1;
  FF im2;
  FF im3;

  static constexpr HOST_DEVICE_INLINE QuartExtensionField zero()
  {
    return QuartExtensionField{FF::zero(), FF::zero(), FF::zero(), FF::zero()};
  }

  static constexpr HOST_DEVICE_INLINE QuartExtensionField one()
  {
    return QuartExtensionField{FF::one(), FF::zero(), FF::zero(), FF::zero()};
  }

  static constexpr HOST_DEVICE_INLINE QuartExtensionField to_montgomery(const QuartExtensionField& xs)
  {
    return QuartExtensionField{
      FF::to_montgomery(xs.real), FF::to_montgomery(xs.im1), FF::to_montgomery(xs.im2), FF::to_montgomery(xs.im3)};
  }

  static constexpr HOST_DEVICE_INLINE QuartExtensionField from_montgomery(const QuartExtensionField& xs)
  {
    return QuartExtensionField{
      FF::from_montgomery(xs.real), FF::from_montgomery(xs.im1), FF::from_montgomery(xs.im2),
      FF::from_montgomery(xs.im3)};
  }

  static HOST_INLINE QuartExtensionField rand_host()
  {
    return QuartExtensionField{FF::rand_host(), FF::rand_host(), FF::rand_host(), FF::rand_host()};
  }

  static void rand_host_many(QuartExtensionField* out, int size)
  {
    for (int i = 0; i < size; i++)
      out[i] = rand_host();
  }

  template <unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE QuartExtensionField sub_modulus(const QuartExtensionField& xs)
  {
    return QuartExtensionField{
      FF::sub_modulus<REDUCTION_SIZE>(&xs.real), FF::sub_modulus<REDUCTION_SIZE>(&xs.im1),
      FF::sub_modulus<REDUCTION_SIZE>(&xs.im2), FF::sub_modulus<REDUCTION_SIZE>(&xs.im3)};
  }

  friend std::ostream& operator<<(std::ostream& os, const QuartExtensionField& xs)
  {
    os << "{ Real: " << xs.real << " }; { Im1: " << xs.im1 << " }; { Im2: " << xs.im2 << " }; { Im3: " << xs.im3
       << " };";
    return os;
  }

  friend HOST_DEVICE_INLINE QuartExtensionField operator+(QuartExtensionField xs, const QuartExtensionField& ys)
  {
    return QuartExtensionField{xs.real + ys.real, xs.im1 + ys.im1, xs.im2 + ys.im2, xs.im3 + ys.im3};
  }

  friend HOST_DEVICE_INLINE QuartExtensionField operator-(QuartExtensionField xs, const QuartExtensionField& ys)
  {
    return QuartExtensionField{xs.real - ys.real, xs.im1 - ys.im1, xs.im2 - ys.im2, xs.im3 - ys.im3};
  }

  friend HOST_DEVICE_INLINE QuartExtensionField operator+(FF xs, const QuartExtensionField& ys)
  {
    return QuartExtensionField{xs + ys.real, ys.im1, ys.im2, ys.im3};
  }

  friend HOST_DEVICE_INLINE QuartExtensionField operator-(FF xs, const QuartExtensionField& ys)
  {
    return QuartExtensionField{xs - ys.real, FF::neg(ys.im1), FF::neg(ys.im2), FF::neg(ys.im3)};
  }

  friend HOST_DEVICE_INLINE QuartExtensionField operator+(QuartExtensionField xs, const FF& ys)
  {
    return QuartExtensionField{xs.real + ys, xs.im1, xs.im2, xs.im3};
  }

  friend HOST_DEVICE_INLINE QuartExtensionField operator-(QuartExtensionField xs, const FF& ys)
  {
    return QuartExtensionField{xs.real - ys, xs.im1, xs.im2, xs.im3};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionWide
  mul_wide(const QuartExtensionField& xs, const QuartExtensionField& ys)
  {
    if (CONFIG::nonresidue_is_negative)
      return ExtensionWide{
        FF::mul_wide(xs.real, ys.real) -
          FF::template mul_unsigned<CONFIG::nonresidue>(
            FF::mul_wide(xs.im1, ys.im3) + FF::mul_wide(xs.im2, ys.im2) + FF::mul_wide(xs.im3, ys.im1)),
        FF::mul_wide(xs.real, ys.im1) + FF::mul_wide(xs.im1, ys.real) -
          FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.im2, ys.im3) + FF::mul_wide(xs.im3, ys.im2)),
        FF::mul_wide(xs.real, ys.im2) + FF::mul_wide(xs.im1, ys.im1) + FF::mul_wide(xs.im2, ys.real) -
          FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.im3, ys.im3)),
        FF::mul_wide(xs.real, ys.im3) + FF::mul_wide(xs.im1, ys.im2) + FF::mul_wide(xs.im2, ys.im1) +
          FF::mul_wide(xs.im3, ys.real)};
    else
      return ExtensionWide{
        FF::mul_wide(xs.real, ys.real) +
          FF::template mul_unsigned<CONFIG::nonresidue>(
            FF::mul_wide(xs.im1, ys.im3) + FF::mul_wide(xs.im2, ys.im2) + FF::mul_wide(xs.im3, ys.im1)),
        FF::mul_wide(xs.real, ys.im1) + FF::mul_wide(xs.im1, ys.real) +
          FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.im2, ys.im3) + FF::mul_wide(xs.im3, ys.im2)),
        FF::mul_wide(xs.real, ys.im2) + FF::mul_wide(xs.im1, ys.im1) + FF::mul_wide(xs.im2, ys.real) +
          FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.im3, ys.im3)),
        FF::mul_wide(xs.real, ys.im3) + FF::mul_wide(xs.im1, ys.im2) + FF::mul_wide(xs.im2, ys.im1) +
          FF::mul_wide(xs.im3, ys.real)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionWide mul_wide(const QuartExtensionField& xs, const FF& ys)
  {
    return ExtensionWide{
      FF::mul_wide(xs.real, ys), FF::mul_wide(xs.im1, ys), FF::mul_wide(xs.im2, ys), FF::mul_wide(xs.im3, ys)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionWide mul_wide(const FF& xs, const QuartExtensionField& ys)
  {
    return ExtensionWide{
      FF::mul_wide(xs, ys.real), FF::mul_wide(xs, ys.im1), FF::mul_wide(xs, ys.im2), FF::mul_wide(xs, ys.im3)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE QuartExtensionField reduce(const ExtensionWide& xs)
  {
    return QuartExtensionField{
      FF::template reduce<MODULUS_MULTIPLE>(xs.real), FF::template reduce<MODULUS_MULTIPLE>(xs.im1),
      FF::template reduce<MODULUS_MULTIPLE>(xs.im2), FF::template reduce<MODULUS_MULTIPLE>(xs.im3)};
  }

  template <class T1, class T2>
  friend HOST_DEVICE_INLINE QuartExtensionField operator*(const T1& xs, const T2& ys)
  {
    ExtensionWide xy = mul_wide(xs, ys);
    return reduce(xy);
  }

  friend HOST_DEVICE_INLINE bool operator==(const QuartExtensionField& xs, const QuartExtensionField& ys)
  {
    return (xs.real == ys.real) && (xs.im1 == ys.im1) && (xs.im2 == ys.im2) && (xs.im3 == ys.im3);
  }

  friend HOST_DEVICE_INLINE bool operator!=(const QuartExtensionField& xs, const QuartExtensionField& ys)
  {
    return !(xs == ys);
  }

  template <uint32_t multiplier, unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE QuartExtensionField mul_unsigned(const QuartExtensionField& xs)
  {
    return {
      FF::template mul_unsigned<multiplier>(xs.real), FF::template mul_unsigned<multiplier>(xs.im1),
      FF::template mul_unsigned<multiplier>(xs.im2), FF::template mul_unsigned<multiplier>(xs.im3)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionWide sqr_wide(const QuartExtensionField& xs)
  {
    // TODO: change to a more efficient squaring
    return mul_wide<MODULUS_MULTIPLE>(xs, xs);
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE QuartExtensionField sqr(const QuartExtensionField& xs)
  {
    // TODO: change to a more efficient squaring
    return xs * xs;
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE QuartExtensionField neg(const QuartExtensionField& xs)
  {
    return {FF::neg(xs.real), FF::neg(xs.im1), FF::neg(xs.im2), FF::neg(xs.im3)};
  }

  // inverse of zero is set to be zero which is what we want most of the time
  static constexpr HOST_DEVICE_INLINE QuartExtensionField inverse(const QuartExtensionField& xs)
  {
    FF x, x0, x2;
    if (CONFIG::nonresidue_is_negative) {
      x0 = FF::reduce(
        FF::sqr_wide(xs.real) +
        FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.im1, xs.im3 + xs.im3) - FF::sqr_wide(xs.im2)));
      x2 = FF::reduce(
        FF::mul_wide(xs.real, xs.im2 + xs.im2) - FF::sqr_wide(xs.im1) +
        FF::template mul_unsigned<CONFIG::nonresidue>(FF::sqr_wide(xs.im3)));
      x = FF::reduce(FF::sqr_wide(x0) + FF::template mul_unsigned<CONFIG::nonresidue>(FF::sqr_wide(x2)));
    } else {
      x0 = FF::reduce(
        FF::sqr_wide(xs.real) -
        FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.im1, xs.im3 + xs.im3) - FF::sqr_wide(xs.im2)));
      x2 = FF::reduce(
        FF::mul_wide(xs.real, xs.im2 + xs.im2) - FF::sqr_wide(xs.im1) -
        FF::template mul_unsigned<CONFIG::nonresidue>(FF::sqr_wide(xs.im3)));
      x = FF::reduce(FF::sqr_wide(x0) - FF::template mul_unsigned<CONFIG::nonresidue>(FF::sqr_wide(x2)));
    }
    FF x_inv = FF::inverse(x);
    x0 = x0 * x_inv;
    x2 = x2 * x_inv;
    return {
      FF::reduce(
        (CONFIG::nonresidue_is_negative
           ? (FF::mul_wide(xs.real, x0) + FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.im2, x2)))
           : (FF::mul_wide(xs.real, x0)) - FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.im2, x2)))),
      FF::reduce(
        (CONFIG::nonresidue_is_negative
           ? FWide::neg(FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.im3, x2)))
           : FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.im3, x2))) -
        FF::mul_wide(xs.im1, x0)),
      FF::reduce(FF::mul_wide(xs.im2, x0) - FF::mul_wide(xs.real, x2)),
      FF::reduce(FF::mul_wide(xs.im1, x2) - FF::mul_wide(xs.im3, x0)),
    };
  }
};

template <class CONFIG, class T>
struct SharedMemory<QuartExtensionField<CONFIG, T>> {
  __device__ QuartExtensionField<CONFIG, T>* getPointer()
  {
    extern __shared__ QuartExtensionField<CONFIG, T> s_ext4_scalar_[];
    return s_ext4_scalar_;
  }
};