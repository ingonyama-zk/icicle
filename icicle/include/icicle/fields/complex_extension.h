#pragma once

#include "field.h"
#include "icicle/utils/modifiers.h"
#ifdef __CUDACC__
#include "gpu-utils/sharedmem.h"
#endif // __CUDACC__

template <typename CONFIG, class T>
class ComplexExtensionField
{
private:
  friend T;

  typedef typename T::Wide FWide;

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
  typedef T FF;
  static constexpr unsigned TLC = 2 * CONFIG::limbs_count;

  FF real;
  FF imaginary;

  static constexpr HOST_DEVICE_INLINE ComplexExtensionField zero()
  {
    return ComplexExtensionField{FF::zero(), FF::zero()};
  }

  static constexpr HOST_DEVICE_INLINE ComplexExtensionField one()
  {
    return ComplexExtensionField{FF::one(), FF::zero()};
  }

  static constexpr HOST_DEVICE_INLINE ComplexExtensionField to_montgomery(const ComplexExtensionField& xs)
  {
    return ComplexExtensionField{xs.real * FF{CONFIG::montgomery_r}, xs.imaginary * FF{CONFIG::montgomery_r}};
  }

  static constexpr HOST_DEVICE_INLINE ComplexExtensionField from_montgomery(const ComplexExtensionField& xs)
  {
    return ComplexExtensionField{xs.real * FF{CONFIG::montgomery_r_inv}, xs.imaginary * FF{CONFIG::montgomery_r_inv}};
  }

  static HOST_INLINE ComplexExtensionField rand_host()
  {
    return ComplexExtensionField{FF::rand_host(), FF::rand_host()};
  }

  static void rand_host_many(ComplexExtensionField* out, int size)
  {
    for (int i = 0; i < size; i++)
      out[i] = rand_host();
  }

  template <unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE ComplexExtensionField sub_modulus(const ComplexExtensionField& xs)
  {
    return ComplexExtensionField{
      FF::sub_modulus<REDUCTION_SIZE>(&xs.real), FF::sub_modulus<REDUCTION_SIZE>(&xs.imaginary)};
  }

  friend std::ostream& operator<<(std::ostream& os, const ComplexExtensionField& xs)
  {
    os << "{ Real: " << xs.real << " }; { Imaginary: " << xs.imaginary << " }";
    return os;
  }

  friend HOST_DEVICE_INLINE ComplexExtensionField operator+(ComplexExtensionField xs, const ComplexExtensionField& ys)
  {
    return ComplexExtensionField{xs.real + ys.real, xs.imaginary + ys.imaginary};
  }

  friend HOST_DEVICE_INLINE ComplexExtensionField operator-(ComplexExtensionField xs, const ComplexExtensionField& ys)
  {
    return ComplexExtensionField{xs.real - ys.real, xs.imaginary - ys.imaginary};
  }

  friend HOST_DEVICE_INLINE ComplexExtensionField operator+(FF xs, const ComplexExtensionField& ys)
  {
    return ComplexExtensionField{xs + ys.real, ys.imaginary};
  }

  friend HOST_DEVICE_INLINE ComplexExtensionField operator-(FF xs, const ComplexExtensionField& ys)
  {
    return ComplexExtensionField{xs - ys.real, FF::neg(ys.imaginary)};
  }

  friend HOST_DEVICE_INLINE ComplexExtensionField operator+(ComplexExtensionField xs, const FF& ys)
  {
    return ComplexExtensionField{xs.real + ys, xs.imaginary};
  }

  friend HOST_DEVICE_INLINE ComplexExtensionField operator-(ComplexExtensionField xs, const FF& ys)
  {
    return ComplexExtensionField{xs.real - ys, xs.imaginary};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionWide
  mul_wide(const ComplexExtensionField& xs, const ComplexExtensionField& ys)
  {
    FWide real_prod = FF::mul_wide(xs.real, ys.real);
    FWide imaginary_prod = FF::mul_wide(xs.imaginary, ys.imaginary);
    FWide prod_of_sums = FF::mul_wide(xs.real + xs.imaginary, ys.real + ys.imaginary);
    FWide nonresidue_times_im = FF::template mul_unsigned<CONFIG::nonresidue>(imaginary_prod);
    nonresidue_times_im = CONFIG::nonresidue_is_negative ? FWide::neg(nonresidue_times_im) : nonresidue_times_im;
    return ExtensionWide{real_prod + nonresidue_times_im, prod_of_sums - real_prod - imaginary_prod};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionWide mul_wide(const ComplexExtensionField& xs, const FF& ys)
  {
    return ExtensionWide{FF::mul_wide(xs.real, ys), FF::mul_wide(xs.imaginary, ys)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionWide mul_wide(const FF& xs, const ComplexExtensionField& ys)
  {
    return mul_wide(ys, xs);
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ComplexExtensionField reduce(const ExtensionWide& xs)
  {
    return ComplexExtensionField{
      FF::template reduce<MODULUS_MULTIPLE>(xs.real), FF::template reduce<MODULUS_MULTIPLE>(xs.imaginary)};
  }

  template <class T1, class T2>
  friend HOST_DEVICE_INLINE ComplexExtensionField operator*(const T1& xs, const T2& ys)
  {
    ExtensionWide xy = mul_wide(xs, ys);
    return reduce(xy);
  }

  friend HOST_DEVICE_INLINE bool operator==(const ComplexExtensionField& xs, const ComplexExtensionField& ys)
  {
    return (xs.real == ys.real) && (xs.imaginary == ys.imaginary);
  }

  friend HOST_DEVICE_INLINE bool operator!=(const ComplexExtensionField& xs, const ComplexExtensionField& ys)
  {
    return !(xs == ys);
  }

  template <const ComplexExtensionField& multiplier>
  static HOST_DEVICE_INLINE ComplexExtensionField mul_const(const ComplexExtensionField& xs)
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
    return ComplexExtensionField{real_prod + nonresidue_times_im, re_im + im_re};
  }

  template <uint32_t multiplier, unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE ComplexExtensionField mul_unsigned(const ComplexExtensionField& xs)
  {
    return {FF::template mul_unsigned<multiplier>(xs.real), FF::template mul_unsigned<multiplier>(xs.imaginary)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ExtensionWide sqr_wide(const ComplexExtensionField& xs)
  {
    // TODO: change to a more efficient squaring
    return mul_wide<MODULUS_MULTIPLE>(xs, xs);
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ComplexExtensionField sqr(const ComplexExtensionField& xs)
  {
    // TODO: change to a more efficient squaring
    return xs * xs;
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ComplexExtensionField neg(const ComplexExtensionField& xs)
  {
    return ComplexExtensionField{FF::neg(xs.real), FF::neg(xs.imaginary)};
  }

  // inverse of zero is set to be zero which is what we want most of the time
  static constexpr HOST_DEVICE_INLINE ComplexExtensionField inverse(const ComplexExtensionField& xs)
  {
    ComplexExtensionField xs_conjugate = {xs.real, FF::neg(xs.imaginary)};
    FF nonresidue_times_im = FF::template mul_unsigned<CONFIG::nonresidue>(FF::sqr(xs.imaginary));
    nonresidue_times_im = CONFIG::nonresidue_is_negative ? FF::neg(nonresidue_times_im) : nonresidue_times_im;
    // TODO: wide here
    FF xs_norm_squared = FF::sqr(xs.real) - nonresidue_times_im;
    return xs_conjugate * ComplexExtensionField{FF::inverse(xs_norm_squared), FF::zero()};
  }
};

#ifdef __CUDACC__
template <typename CONFIG, class T>
struct SharedMemory<ComplexExtensionField<CONFIG, T>> {
  __device__ ComplexExtensionField<CONFIG, T>* getPointer()
  {
    extern __shared__ ComplexExtensionField<CONFIG, T> s_ext2_scalar_[];
    return s_ext2_scalar_;
  }
};
#endif //__CUDAC__
