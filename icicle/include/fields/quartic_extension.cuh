#pragma once

#include "field.cuh"
#include "../gpu-utils/modifiers.cuh"
#include "../gpu-utils/sharedmem.cuh"

template <typename CONFIG>
class ExtensionField
{
private:
  typedef typename Field<CONFIG>::Wide FWide;

  struct ExtensionWide {
    FWide real;
    FWide im1;
    FWide im2;
    FWide im3;

    friend ExtensionWide operator+(ExtensionWide xs, const ExtensionWide& ys)
    {
      return ExtensionWide{xs.real + ys.real, xs.im1 + ys.im1, xs.im2 + ys.im2, xs.im3 + ys.im3};
    }

    friend ExtensionWide operator-(ExtensionWide xs, const ExtensionWide& ys)
    {
      return ExtensionWide{xs.real - ys.real, xs.im1 - ys.im1, xs.im2 - ys.im2, xs.im3 - ys.im3};
    }
  };

public:
  typedef Field<CONFIG> FF;
  static constexpr unsigned TLC = 4 * CONFIG::limbs_count;

  FF real;
  FF im1;
  FF im2;
  FF im3;

  static constexpr ExtensionField zero()
  {
    return ExtensionField{FF::zero(), FF::zero(), FF::zero(), FF::zero()};
  }

  static constexpr ExtensionField one()
  {
    return ExtensionField{FF::one(), FF::zero(), FF::zero(), FF::zero()};
  }

  static constexpr ExtensionField to_montgomery(const ExtensionField& xs)
  {
    return ExtensionField{
      xs.real * FF{CONFIG::montgomery_r}, xs.im1 * FF{CONFIG::montgomery_r}, xs.im2 * FF{CONFIG::montgomery_r},
      xs.im3 * FF{CONFIG::montgomery_r}};
  }

  static constexpr ExtensionField from_montgomery(const ExtensionField& xs)
  {
    return ExtensionField{
      xs.real * FF{CONFIG::montgomery_r_inv}, xs.im1 * FF{CONFIG::montgomery_r_inv},
      xs.im2 * FF{CONFIG::montgomery_r_inv}, xs.im3 * FF{CONFIG::montgomery_r_inv}};
  }

  static ExtensionField rand_host()
  {
    return ExtensionField{FF::rand_host(), FF::rand_host(), FF::rand_host(), FF::rand_host()};
  }

  static void rand_host_many(ExtensionField* out, int size)
  {
    for (int i = 0; i < size; i++)
      out[i] = rand_host();
  }

  template <unsigned REDUCTION_SIZE = 1>
  static constexpr ExtensionField sub_modulus(const ExtensionField& xs)
  {
    return ExtensionField{
      FF::sub_modulus<REDUCTION_SIZE>(&xs.real), FF::sub_modulus<REDUCTION_SIZE>(&xs.im1),
      FF::sub_modulus<REDUCTION_SIZE>(&xs.im2), FF::sub_modulus<REDUCTION_SIZE>(&xs.im3)};
  }

  friend std::ostream& operator<<(std::ostream& os, const ExtensionField& xs)
  {
    os << "{ Real: " << xs.real << " }; { Im1: " << xs.im1 << " }; { Im2: " << xs.im2 << " }; { Im3: " << xs.im3
       << " };";
    return os;
  }

  friend ExtensionField operator+(ExtensionField xs, const ExtensionField& ys)
  {
    return ExtensionField{xs.real + ys.real, xs.im1 + ys.im1, xs.im2 + ys.im2, xs.im3 + ys.im3};
  }

  friend ExtensionField operator-(ExtensionField xs, const ExtensionField& ys)
  {
    return ExtensionField{xs.real - ys.real, xs.im1 - ys.im1, xs.im2 - ys.im2, xs.im3 - ys.im3};
  }

  friend ExtensionField operator+(FF xs, const ExtensionField& ys)
  {
    return ExtensionField{xs + ys.real, ys.im1, ys.im2, ys.im3};
  }

  friend ExtensionField operator-(FF xs, const ExtensionField& ys)
  {
    return ExtensionField{xs - ys.real, FF::neg(ys.im1), FF::neg(ys.im2), FF::neg(ys.im3)};
  }

  friend ExtensionField operator+(ExtensionField xs, const FF& ys)
  {
    return ExtensionField{xs.real + ys, xs.im1, xs.im2, xs.im3};
  }

  friend ExtensionField operator-(ExtensionField xs, const FF& ys)
  {
    return ExtensionField{xs.real - ys, xs.im1, xs.im2, xs.im3};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr ExtensionWide mul_wide(const ExtensionField& xs, const ExtensionField& ys)
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
  static constexpr ExtensionWide mul_wide(const ExtensionField& xs, const FF& ys)
  {
    return ExtensionWide{
      FF::mul_wide(xs.real, ys), FF::mul_wide(xs.im1, ys), FF::mul_wide(xs.im2, ys), FF::mul_wide(xs.im3, ys)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr ExtensionWide mul_wide(const FF& xs, const ExtensionField& ys)
  {
    return ExtensionWide{
      FF::mul_wide(xs, ys.real), FF::mul_wide(xs, ys.im1), FF::mul_wide(xs, ys.im2), FF::mul_wide(xs, ys.im3)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr ExtensionField reduce(const ExtensionWide& xs)
  {
    return ExtensionField{
      FF::template reduce<MODULUS_MULTIPLE>(xs.real), FF::template reduce<MODULUS_MULTIPLE>(xs.im1),
      FF::template reduce<MODULUS_MULTIPLE>(xs.im2), FF::template reduce<MODULUS_MULTIPLE>(xs.im3)};
  }

  template <class T1, class T2>
  friend ExtensionField operator*(const T1& xs, const T2& ys)
  {
    ExtensionWide xy = mul_wide(xs, ys);
    return reduce(xy);
  }

  friend bool operator==(const ExtensionField& xs, const ExtensionField& ys)
  {
    return (xs.real == ys.real) && (xs.im1 == ys.im1) && (xs.im2 == ys.im2) && (xs.im3 == ys.im3);
  }

  friend bool operator!=(const ExtensionField& xs, const ExtensionField& ys) { return !(xs == ys); }

  template <uint32_t multiplier, unsigned REDUCTION_SIZE = 1>
  static constexpr ExtensionField mul_unsigned(const ExtensionField& xs)
  {
    return {
      FF::template mul_unsigned<multiplier>(xs.real), FF::template mul_unsigned<multiplier>(xs.im1),
      FF::template mul_unsigned<multiplier>(xs.im2), FF::template mul_unsigned<multiplier>(xs.im3)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr ExtensionWide sqr_wide(const ExtensionField& xs)
  {
    // TODO: change to a more efficient squaring
    return mul_wide<MODULUS_MULTIPLE>(xs, xs);
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr ExtensionField sqr(const ExtensionField& xs)
  {
    // TODO: change to a more efficient squaring
    return xs * xs;
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr ExtensionField neg(const ExtensionField& xs)
  {
    return {FF::neg(xs.real), FF::neg(xs.im1), FF::neg(xs.im2), FF::neg(xs.im3)};
  }

  // inverse of zero is set to be zero which is what we want most of the time
  static constexpr ExtensionField inverse(const ExtensionField& xs)
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

template <class CONFIG>
struct SharedMemory<ExtensionField<CONFIG>> {
  ExtensionField<CONFIG>* getPointer()
  {
    ExtensionField<CONFIG> *s_ext4_scalar_=nullptr;
    return s_ext4_scalar_;
  }
};