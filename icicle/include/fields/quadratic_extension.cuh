#pragma once

#include "field.cuh"

#define HOST_INLINE        __host__ __forceinline__
#define DEVICE_INLINE      __device__ __forceinline__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__

template <typename CONFIG>
class Extension2Field
{
private:
  typedef typename Field<CONFIG>::Wide FWide;

  struct Extension2Wide {
    FWide real;
    FWide imaginary;

    friend HOST_DEVICE_INLINE Extension2Wide operator+(Extension2Wide xs, const Extension2Wide& ys)
    {
      return Extension2Wide{xs.real + ys.real, xs.imaginary + ys.imaginary};
    }

    friend HOST_DEVICE_INLINE Extension2Wide operator-(Extension2Wide xs, const Extension2Wide& ys)
    {
      return Extension2Wide{xs.real - ys.real, xs.imaginary - ys.imaginary};
    }
  };

public:
  typedef Field<CONFIG> FF;
  static constexpr unsigned TLC = 2 * CONFIG::limbs_count;

  FF real;
  FF imaginary;

  static constexpr HOST_DEVICE_INLINE Extension2Field zero() { return Extension2Field{FF::zero(), FF::zero()}; }

  static constexpr HOST_DEVICE_INLINE Extension2Field one() { return Extension2Field{FF::one(), FF::zero()}; }

  static constexpr HOST_DEVICE_INLINE Extension2Field ToMontgomery(const Extension2Field& xs)
  {
    return Extension2Field{xs.real * FF{CONFIG::montgomery_r}, xs.imaginary * FF{CONFIG::montgomery_r}};
  }

  static constexpr HOST_DEVICE_INLINE Extension2Field FromMontgomery(const Extension2Field& xs)
  {
    return Extension2Field{xs.real * FF{CONFIG::montgomery_r_inv}, xs.imaginary * FF{CONFIG::montgomery_r_inv}};
  }

  static HOST_INLINE Extension2Field rand_host() { return Extension2Field{FF::rand_host(), FF::rand_host()}; }

  template <unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE Extension2Field sub_modulus(const Extension2Field& xs)
  {
    return Extension2Field{FF::sub_modulus<REDUCTION_SIZE>(&xs.real), FF::sub_modulus<REDUCTION_SIZE>(&xs.imaginary)};
  }

  friend std::ostream& operator<<(std::ostream& os, const Extension2Field& xs)
  {
    os << "{ Real: " << xs.real << " }; { Imaginary: " << xs.imaginary << " }";
    return os;
  }

  friend HOST_DEVICE_INLINE Extension2Field operator+(Extension2Field xs, const Extension2Field& ys)
  {
    return Extension2Field{xs.real + ys.real, xs.imaginary + ys.imaginary};
  }

  friend HOST_DEVICE_INLINE Extension2Field operator-(Extension2Field xs, const Extension2Field& ys)
  {
    return Extension2Field{xs.real - ys.real, xs.imaginary - ys.imaginary};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Extension2Wide mul_wide(const Extension2Field& xs, const Extension2Field& ys)
  {
    FWide real_prod = FF::mul_wide(xs.real, ys.real);
    FWide imaginary_prod = FF::mul_wide(xs.imaginary, ys.imaginary);
    FWide prod_of_sums = FF::mul_wide(xs.real + xs.imaginary, ys.real + ys.imaginary);
    FWide nonresidue_times_im = FF::template mul_unsigned<CONFIG::nonresidue>(imaginary_prod);
    nonresidue_times_im = CONFIG::nonresidue_is_negative ? FWide::neg(nonresidue_times_im) : nonresidue_times_im;
    return Extension2Wide{real_prod + nonresidue_times_im, prod_of_sums - real_prod - imaginary_prod};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Extension2Field reduce(const Extension2Wide& xs)
  {
    return Extension2Field{
      FF::template reduce<MODULUS_MULTIPLE>(xs.real), FF::template reduce<MODULUS_MULTIPLE>(xs.imaginary)};
  }

  friend HOST_DEVICE_INLINE Extension2Field operator*(const Extension2Field& xs, const Extension2Field& ys)
  {
    Extension2Wide xy = mul_wide(xs, ys);
    return reduce(xy);
  }

  friend HOST_DEVICE_INLINE bool operator==(const Extension2Field& xs, const Extension2Field& ys)
  {
    return (xs.real == ys.real) && (xs.imaginary == ys.imaginary);
  }

  friend HOST_DEVICE_INLINE bool operator!=(const Extension2Field& xs, const Extension2Field& ys) { return !(xs == ys); }

  template <const Extension2Field& multiplier>
  static HOST_DEVICE_INLINE Extension2Field mul_const(const Extension2Field& xs)
  {
    static constexpr FF mul_real = multiplier.real;
    static constexpr FF mul_imaginary = multiplier.imaginary;
    const FF xs_real = xs.real;
    const FF xs_imaginary = xs.imaginary;
    FF real_prod = FF::template mul_const<mul_real>(xs_real);
    FF imaginary_prod = FF::template mul_const<mul_imaginary>(xs_imaginary);
    FF re_im = FF::template mul_const<mul_real>(xs_imaginary);
    FF im_re = FF::template mul_const<mul_imaginary>(xs_real);
    FF nonresidue_times_im = FF::template mul_unsigned<CONFIG::nonresidue>(imaginary_prod);
    nonresidue_times_im = CONFIG::nonresidue_is_negative ? FF::neg(nonresidue_times_im) : nonresidue_times_im;
    return Extension2Field{real_prod + nonresidue_times_im, re_im + im_re};
  }

  template <uint32_t multiplier, unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE Extension2Field mul_unsigned(const Extension2Field& xs)
  {
    return {FF::template mul_unsigned<multiplier>(xs.real), FF::template mul_unsigned<multiplier>(xs.imaginary)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Extension2Wide sqr_wide(const Extension2Field& xs)
  {
    // TODO: change to a more efficient squaring
    return mul_wide<MODULUS_MULTIPLE>(xs, xs);
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Extension2Field sqr(const Extension2Field& xs)
  {
    // TODO: change to a more efficient squaring
    return xs * xs;
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Extension2Field neg(const Extension2Field& xs)
  {
    return Extension2Field{FF::neg(xs.real), FF::neg(xs.imaginary)};
  }

  // inverse of zero is set to be zero which is what we want most of the time
  static constexpr HOST_DEVICE_INLINE Extension2Field inverse(const Extension2Field& xs)
  {
    Extension2Field xs_conjugate = {xs.real, FF::neg(xs.imaginary)};
    FF nonresidue_times_im = FF::template mul_unsigned<CONFIG::nonresidue>(FF::sqr(xs.imaginary));
    nonresidue_times_im = CONFIG::nonresidue_is_negative ? FF::neg(nonresidue_times_im) : nonresidue_times_im;
    // TODO: wide here
    FF xs_norm_squared = FF::sqr(xs.real) - nonresidue_times_im;
    return xs_conjugate * Extension2Field{FF::inverse(xs_norm_squared), FF::zero()};
  }
};