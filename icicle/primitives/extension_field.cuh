#pragma once

#include "field.cuh"

#define HOST_INLINE        __host__ __forceinline__
#define DEVICE_INLINE      __device__ __forceinline__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__

template <typename CONFIG>
class ExtensionField
{
private:
  typedef typename Field<CONFIG>::Wide FWide;

  struct ExtensionWide {
    FWide real;
    FWide imaginary;

    friend HOST_DEVICE_INLINE ExtensionWide operator+(ExtensionWide xs, const ExtensionWide& ys)
    {
      return ExtensionWide{xs.real + ys.real, xs.imaginary + ys.imaginary};
    }

    friend HOST_DEVICE_INLINE ExtensionWide operator-(ExtensionWide xs, const ExtensionWide& ys)
    {
      return ExtensionWide{xs.real - ys.real, xs.imaginary - ys.imaginary};
    }
  };

public:
  typedef Field<CONFIG> FF;
  static constexpr unsigned TLC = 2 * CONFIG::limbs_count;

  FF real;
  FF imaginary;

  static constexpr HOST_DEVICE_INLINE ExtensionField zero() { return ExtensionField{FF::zero(), FF::zero()}; }

  static constexpr HOST_DEVICE_INLINE ExtensionField one() { return ExtensionField{FF::one(), FF::zero()}; }

  static constexpr HOST_DEVICE_INLINE ExtensionField ToMontgomery(const ExtensionField& xs)
  {
    return ExtensionField{xs.real * FF{CONFIG::montgomery_r}, xs.imaginary * FF{CONFIG::montgomery_r}};
  }

  static constexpr HOST_DEVICE_INLINE ExtensionField FromMontgomery(const ExtensionField& xs)
  {
    return ExtensionField{xs.real * FF{CONFIG::montgomery_r_inv}, xs.imaginary * FF{CONFIG::montgomery_r_inv}};
  }

  static HOST_INLINE ExtensionField rand_host() { return ExtensionField{FF::rand_host(), FF::rand_host()}; }

  template <unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionField sub_modulus(const ExtensionField& xs)
  {
    return ExtensionField{FF::sub_modulus<REDUCTION_SIZE>(&xs.real), FF::sub_modulus<REDUCTION_SIZE>(&xs.imaginary)};
  }

  friend std::ostream& operator<<(std::ostream& os, const ExtensionField& xs)
  {
    os << "{ Real: " << xs.real << " }; { Imaginary: " << xs.imaginary << " }";
    return os;
  }

  friend HOST_DEVICE_INLINE ExtensionField operator+(ExtensionField xs, const ExtensionField& ys)
  {
    return ExtensionField{xs.real + ys.real, xs.imaginary + ys.imaginary};
  }

  friend HOST_DEVICE_INLINE ExtensionField operator-(ExtensionField xs, const ExtensionField& ys)
  {
    return ExtensionField{xs.real - ys.real, xs.imaginary - ys.imaginary};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionWide mul_wide(const ExtensionField& xs, const ExtensionField& ys)
  {
    FWide real_prod = FF::mul_wide(xs.real, ys.real);
    FWide imaginary_prod = FF::mul_wide(xs.imaginary, ys.imaginary);
    FWide prod_of_sums = FF::mul_wide(xs.real + xs.imaginary, ys.real + ys.imaginary);
    FWide i_sq_times_im = FF::template mul_unsigned<CONFIG::i_squared>(imaginary_prod);
    i_sq_times_im = CONFIG::i_squared_is_negative ? FWide::neg(i_sq_times_im) : i_sq_times_im;
    return ExtensionWide{real_prod + i_sq_times_im, prod_of_sums - real_prod - imaginary_prod};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionField reduce(const ExtensionWide& xs)
  {
    return ExtensionField{
      FF::template reduce<MODULUS_MULTIPLE>(xs.real), FF::template reduce<MODULUS_MULTIPLE>(xs.imaginary)};
  }

  friend HOST_DEVICE_INLINE ExtensionField operator*(const ExtensionField& xs, const ExtensionField& ys)
  {
    ExtensionWide xy = mul_wide(xs, ys);
    return reduce(xy);
  }

  friend HOST_DEVICE_INLINE bool operator==(const ExtensionField& xs, const ExtensionField& ys)
  {
    return (xs.real == ys.real) && (xs.imaginary == ys.imaginary);
  }

  friend HOST_DEVICE_INLINE bool operator!=(const ExtensionField& xs, const ExtensionField& ys) { return !(xs == ys); }

  template <const ExtensionField& multiplier>
  static HOST_DEVICE_INLINE ExtensionField mul_const(const ExtensionField& xs)
  {
    static constexpr FF mul_real = multiplier.real;
    static constexpr FF mul_imaginary = multiplier.imaginary;
    const FF xs_real = xs.real;
    const FF xs_imaginary = xs.imaginary;
    FF real_prod = FF::template mul_const<mul_real>(xs_real);
    FF imaginary_prod = FF::template mul_const<mul_imaginary>(xs_imaginary);
    FF re_im = FF::template mul_const<mul_real>(xs_imaginary);
    FF im_re = FF::template mul_const<mul_imaginary>(xs_real);
    FF i_sq_times_im = FF::template mul_unsigned<CONFIG::i_squared>(imaginary_prod);
    i_sq_times_im = CONFIG::i_squared_is_negative ? FF::neg(i_sq_times_im) : i_sq_times_im;
    return ExtensionField{real_prod + i_sq_times_im, re_im + im_re};
  }

  template <uint32_t multiplier, unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionField mul_unsigned(const ExtensionField& xs)
  {
    return {FF::template mul_unsigned<multiplier>(xs.real), FF::template mul_unsigned<multiplier>(xs.imaginary)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionWide sqr_wide(const ExtensionField& xs)
  {
    // TODO: change to a more efficient squaring
    return mul_wide<MODULUS_MULTIPLE>(xs, xs);
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionField sqr(const ExtensionField& xs)
  {
    // TODO: change to a more efficient squaring
    return xs * xs;
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionField neg(const ExtensionField& xs)
  {
    return ExtensionField{FF::neg(xs.real), FF::neg(xs.imaginary)};
  }

  // inverse assumes that xs is nonzero
  static constexpr HOST_DEVICE_INLINE ExtensionField inverse(const ExtensionField& xs)
  {
    ExtensionField xs_conjugate = {xs.real, FF::neg(xs.imaginary)};
    FF i_sq_times_im = FF::template mul_unsigned<CONFIG::i_squared>(FF::sqr(xs.imaginary));
    i_sq_times_im = CONFIG::i_squared_is_negative ? FF::neg(i_sq_times_im) : i_sq_times_im;
    // TODO: wide here
    FF xs_norm_squared = FF::sqr(xs.real) - i_sq_times_im;
    return xs_conjugate * ExtensionField{FF::inverse(xs_norm_squared), FF::zero()};
  }
};