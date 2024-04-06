#pragma once

#include "field.cuh"

template <typename CONFIG>
class Extension4Field
{
private:
  typedef typename Field<CONFIG>::Wide FWide;

  struct Extension4Wide {
    FWide real;
    FWide im1;
    FWide im2;
    FWide im3;

    friend HOST_DEVICE_INLINE Extension4Wide operator+(Extension4Wide xs, const Extension4Wide& ys)
    {
      return Extension4Wide{xs.real + ys.real, xs.im1 + ys.im1, xs.im2 + ys.im2, xs.im3 + ys.im3};
    }

    friend HOST_DEVICE_INLINE Extension4Wide operator-(Extension4Wide xs, const Extension4Wide& ys)
    {
      return Extension4Wide{xs.real - ys.real, xs.im1 - ys.im1, xs.im2 - ys.im2, xs.im3 - ys.im3};
    }
  };

public:
  typedef Field<CONFIG> FF;
  static constexpr unsigned TLC = 4 * CONFIG::limbs_count;

  FF real;
  FF im1;
  FF im2;
  FF im3;

  static constexpr HOST_DEVICE_INLINE Extension4Field zero()
  {
    return Extension4Field{FF::zero(), FF::zero(), FF::zero(), FF::zero()};
  }

  static constexpr HOST_DEVICE_INLINE Extension4Field one()
  {
    return Extension4Field{FF::one(), FF::zero(), FF::zero(), FF::zero()};
  }

  static constexpr HOST_DEVICE_INLINE Extension4Field ToMontgomery(const Extension4Field& xs)
  {
    return Extension4Field{
      xs.real * FF{CONFIG::montgomery_r}, xs.im1 * FF{CONFIG::montgomery_r},
      xs.im2 * FF{CONFIG::montgomery_r}, xs.im3 * FF{CONFIG::montgomery_r}};
  }

  static constexpr HOST_DEVICE_INLINE Extension4Field FromMontgomery(const Extension4Field& xs)
  {
    return Extension4Field{
      xs.real * FF{CONFIG::montgomery_r_inv}, xs.im1 * FF{CONFIG::montgomery_r_inv},
      xs.im2 * FF{CONFIG::montgomery_r_inv}, xs.im3 * FF{CONFIG::montgomery_r_inv}};
  }

  static HOST_INLINE Extension4Field rand_host()
  {
    return Extension4Field{FF::rand_host(), FF::rand_host(), FF::rand_host(), FF::rand_host()};
  }

  template <unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE Extension4Field sub_modulus(const Extension4Field& xs)
  {
    return Extension4Field{
      FF::sub_modulus<REDUCTION_SIZE>(&xs.real), FF::sub_modulus<REDUCTION_SIZE>(&xs.im1),
      FF::sub_modulus<REDUCTION_SIZE>(&xs.im2), FF::sub_modulus<REDUCTION_SIZE>(&xs.im3)};
  }

  friend std::ostream& operator<<(std::ostream& os, const Extension4Field& xs)
  {
    os << "{ Real: " << xs.real << " }; { Im1: " << xs.im1 << " }; { Im2: " << xs.im2 << " }; { Im3: " << xs.im3
       << " };";
    return os;
  }

  friend HOST_DEVICE_INLINE Extension4Field operator+(Extension4Field xs, const Extension4Field& ys)
  {
    return Extension4Field{xs.real + ys.real, xs.im1 + ys.im1, xs.im2 + ys.im2, xs.im3 + ys.im3};
  }

  friend HOST_DEVICE_INLINE Extension4Field operator-(Extension4Field xs, const Extension4Field& ys)
  {
    return Extension4Field{xs.real - ys.real, xs.im1 - ys.im1, xs.im2 - ys.im2, xs.im3 - ys.im3};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Extension4Wide mul_wide(const Extension4Field& xs, const Extension4Field& ys)
  {
    if (CONFIG::nonresidue_is_negative)
      return Extension4Wide{
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
      return Extension4Wide{
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
  static constexpr HOST_DEVICE_INLINE Extension4Field reduce(const Extension4Wide& xs)
  {
    return Extension4Field{
      FF::template reduce<MODULUS_MULTIPLE>(xs.real), FF::template reduce<MODULUS_MULTIPLE>(xs.im1),
      FF::template reduce<MODULUS_MULTIPLE>(xs.im2), FF::template reduce<MODULUS_MULTIPLE>(xs.im3)};
  }

  friend HOST_DEVICE_INLINE Extension4Field operator*(const Extension4Field& xs, const Extension4Field& ys)
  {
    Extension4Wide xy = mul_wide(xs, ys);
    return reduce(xy);
  }

  friend HOST_DEVICE_INLINE bool operator==(const Extension4Field& xs, const Extension4Field& ys)
  {
    return (xs.real == ys.real) && (xs.im1 == ys.im1) && (xs.im2 == ys.im2) && (xs.im3 == ys.im3);
  }

  friend HOST_DEVICE_INLINE bool operator!=(const Extension4Field& xs, const Extension4Field& ys)
  {
    return !(xs == ys);
  }

  template <uint32_t multiplier, unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE Extension4Field mul_unsigned(const Extension4Field& xs)
  {
    return {
      FF::template mul_unsigned<multiplier>(xs.real), FF::template mul_unsigned<multiplier>(xs.im1),
      FF::template mul_unsigned<multiplier>(xs.im2), FF::template mul_unsigned<multiplier>(xs.im3)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Extension4Wide sqr_wide(const Extension4Field& xs)
  {
    // TODO: change to a more efficient squaring
    return mul_wide<MODULUS_MULTIPLE>(xs, xs);
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Extension4Field sqr(const Extension4Field& xs)
  {
    // TODO: change to a more efficient squaring
    return xs * xs;
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Extension4Field neg(const Extension4Field& xs)
  {
    return {FF::neg(xs.real), FF::neg(xs.im1), FF::neg(xs.im2), FF::neg(xs.im3)};
  }

  // inverse of zero is set to be zero which is what we want most of the time
  static constexpr HOST_DEVICE_INLINE Extension4Field inverse(const Extension4Field& xs)
  {
    FF x, x0, x2;
    if (CONFIG::nonresidue_is_negative) {
      x0 = FF::reduce(FF::sqr_wide(xs.real) + FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.im1, xs.im3 + xs.im3) - FF::sqr_wide(xs.im2)));
      x2 = FF::reduce(FF::mul_wide(xs.real, xs.im2 + xs.im2) - FF::sqr_wide(xs.im1) + FF::template mul_unsigned<CONFIG::nonresidue>(FF::sqr_wide(xs.im3)));
      x = FF::reduce(FF::sqr_wide(x0) + FF::template mul_unsigned<CONFIG::nonresidue>(FF::sqr_wide(x2)));
    } else {
      x0 = FF::reduce(FF::sqr_wide(xs.real) - FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.im1, xs.im3 + xs.im3) - FF::sqr_wide(xs.im2)));
      x2 = FF::reduce(FF::mul_wide(xs.real, xs.im2 + xs.im2) - FF::sqr_wide(xs.im1) - FF::template mul_unsigned<CONFIG::nonresidue>(FF::sqr_wide(xs.im3)));
      x = FF::reduce(FF::sqr_wide(x0) - FF::template mul_unsigned<CONFIG::nonresidue>(FF::sqr_wide(x2)));
    }
    FF x_inv = FF::inverse(x);
    x0 = x0 * x_inv;
    x2 = x2 * x_inv;
    return {
      FF::reduce((CONFIG::nonresidue_is_negative ? (FF::mul_wide(xs.real, x0) + FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.im2, x2))) : (FF::mul_wide(xs.real, x0)) - FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.im2, x2)))),
      FF::reduce((CONFIG::nonresidue_is_negative ? FWide::neg(FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.im3, x2))) : FF::template mul_unsigned<CONFIG::nonresidue>(FF::mul_wide(xs.im3, x2))) - FF::mul_wide(xs.im1, x0)),
      FF::reduce(FF::mul_wide(xs.im2, x0) - FF::mul_wide(xs.real, x2)),
      FF::reduce(FF::mul_wide(xs.im1, x2) - FF::mul_wide(xs.im3, x0)),
    };
  }
};