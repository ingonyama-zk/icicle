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

public:
  struct Wide {
    FWide c0;
    FWide c1;

    static constexpr Wide HOST_DEVICE_INLINE from_field(const ComplexExtensionField& xs)
    {
      return Wide{FWide::from_field(xs.c0), FWide::from_field(xs.c1)};
    }

    friend HOST_DEVICE_INLINE Wide operator+(const Wide& xs, const Wide& ys)
    {
      return Wide{xs.c0 + ys.c0, xs.c1 + ys.c1};
    }

    friend HOST_DEVICE_INLINE Wide operator-(const Wide& xs, const Wide& ys)
    {
      return Wide{xs.c0 - ys.c0, xs.c1 - ys.c1};
    }

    static constexpr HOST_DEVICE_INLINE Wide neg(const Wide& xs) { return Wide{FWide::neg(xs.c0), FWide::neg(xs.c1)}; }
  };

  typedef T FF;
  static constexpr unsigned TLC = 2 * FF::TLC;

  FF c0;
  FF c1;

  static constexpr HOST_DEVICE_INLINE ComplexExtensionField zero()
  {
    return ComplexExtensionField{FF::zero(), FF::zero()};
  }

  static constexpr HOST_DEVICE_INLINE ComplexExtensionField one()
  {
    return ComplexExtensionField{FF::one(), FF::zero()};
  }

  // Converts a uint32_t value to a QuarticExtensionField element.
  // If `val` ≥ p, it wraps around modulo p, affecting only the first coefficient.
  static constexpr HOST_DEVICE_INLINE ComplexExtensionField from(uint32_t val)
  {
    return ComplexExtensionField{FF::from(val), FF::zero()};
  }

  static constexpr HOST_DEVICE_INLINE ComplexExtensionField to_montgomery(const ComplexExtensionField& xs)
  {
    return ComplexExtensionField{FF::to_montgomery(xs.c0), FF::to_montgomery(xs.c1)};
  }

  static constexpr HOST_DEVICE_INLINE ComplexExtensionField from_montgomery(const ComplexExtensionField& xs)
  {
    return ComplexExtensionField{FF::from_montgomery(xs.c0), FF::from_montgomery(xs.c1)};
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
    return ComplexExtensionField{FF::sub_modulus<REDUCTION_SIZE>(&xs.c0), FF::sub_modulus<REDUCTION_SIZE>(&xs.c1)};
  }

  friend std::ostream& operator<<(std::ostream& os, const ComplexExtensionField& xs)
  {
    os << "{ Real: " << xs.c0 << " }; { Imaginary: " << xs.c1 << " }";
    return os;
  }

  friend HOST_DEVICE_INLINE ComplexExtensionField operator+(ComplexExtensionField xs, const ComplexExtensionField& ys)
  {
    return ComplexExtensionField{xs.c0 + ys.c0, xs.c1 + ys.c1};
  }

  friend HOST_DEVICE_INLINE ComplexExtensionField operator-(ComplexExtensionField xs, const ComplexExtensionField& ys)
  {
    return ComplexExtensionField{xs.c0 - ys.c0, xs.c1 - ys.c1};
  }

  friend HOST_DEVICE_INLINE ComplexExtensionField operator+(FF xs, const ComplexExtensionField& ys)
  {
    return ComplexExtensionField{xs + ys.c0, ys.c1};
  }

  friend HOST_DEVICE_INLINE ComplexExtensionField operator-(FF xs, const ComplexExtensionField& ys)
  {
    return ComplexExtensionField{xs - ys.c0, FF::neg(ys.c1)};
  }

  friend HOST_DEVICE_INLINE ComplexExtensionField operator+(ComplexExtensionField xs, const FF& ys)
  {
    return ComplexExtensionField{xs.c0 + ys, xs.c1};
  }

  friend HOST_DEVICE_INLINE ComplexExtensionField operator-(ComplexExtensionField xs, const FF& ys)
  {
    return ComplexExtensionField{xs.c0 - ys, xs.c1};
  }

  constexpr HOST_DEVICE_INLINE ComplexExtensionField operator-() const
  {
    return ComplexExtensionField{FF::neg(c0), FF::neg(c1)};
  }

  constexpr HOST_DEVICE_INLINE ComplexExtensionField& operator+=(const ComplexExtensionField& ys)
  {
    *this = *this + ys;
    return *this;
  }

  constexpr HOST_DEVICE_INLINE ComplexExtensionField& operator-=(const ComplexExtensionField& ys)
  {
    *this = *this - ys;
    return *this;
  }

  constexpr HOST_DEVICE_INLINE ComplexExtensionField& operator*=(const ComplexExtensionField& ys)
  {
    *this = *this * ys;
    return *this;
  }

  constexpr HOST_DEVICE_INLINE ComplexExtensionField& operator+=(const FF& ys)
  {
    *this = *this + ys;
    return *this;
  }

  constexpr HOST_DEVICE_INLINE ComplexExtensionField& operator-=(const FF& ys)
  {
    *this = *this - ys;
    return *this;
  }

  constexpr HOST_DEVICE_INLINE ComplexExtensionField& operator*=(const FF& ys)
  {
    *this = *this * ys;
    return *this;
  }

  /**
   * @brief Multiplies a field element by the nonresidue of the field
   */
  static constexpr HOST_DEVICE FF mul_by_nonresidue(const FF& xs)
  {
    if constexpr (CONFIG::nonresidue_is_u32) {
      return FF::template mul_unsigned<CONFIG::nonresidue>(xs);
    } else {
      return FF::template mul_const<CONFIG::nonresidue>(xs);
    }
  }

  /**
   * @brief Multiplies a wide field element by the nonresidue of the field
   */
  static constexpr HOST_DEVICE FWide mul_by_nonresidue(const FWide& xs)
  {
    if constexpr (CONFIG::nonresidue_is_u32) {
      return FF::template mul_unsigned<CONFIG::nonresidue>(xs);
    } else {
      return FF::template mul_wide<>(FF::reduce(xs), CONFIG::nonresidue);
    }
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE Wide mul_wide(const ComplexExtensionField& xs, const ComplexExtensionField& ys)
  {
#ifdef __CUDA_ARCH__
    constexpr bool do_karatsuba = FF::TLC >= 8; // The basic multiplier size is 1 limb, Karatsuba is more efficient when
                                                // the multiplication is 8 times wider or more
#else
    constexpr bool do_karatsuba = FF::TLC >= 16; // The basic multiplier size is 2 limbs, Karatsuba is more efficient
                                                 // when the multiplication is 8 times wider or more
#endif

    if constexpr (do_karatsuba) {
      FWide real_prod = FF::mul_wide(xs.c0, ys.c0);
      FWide imaginary_prod = FF::mul_wide(xs.c1, ys.c1);
      FWide prod_of_sums = FF::mul_wide(xs.c0 + xs.c1, ys.c0 + ys.c1);
      FWide nonresidue_times_im = mul_by_nonresidue(imaginary_prod);
      nonresidue_times_im = CONFIG::nonresidue_is_negative ? FWide::neg(nonresidue_times_im) : nonresidue_times_im;
      return Wide{real_prod + nonresidue_times_im, prod_of_sums - real_prod - imaginary_prod};
    } else {
      FWide real_prod = FF::mul_wide(xs.c0, ys.c0);
      FWide imaginary_prod = FF::mul_wide(xs.c1, ys.c1);
      FWide ab = FF::mul_wide(xs.c0, ys.c1);
      FWide ba = FF::mul_wide(xs.c1, ys.c0);
      FWide nonresidue_times_im = mul_by_nonresidue(imaginary_prod);
      nonresidue_times_im = CONFIG::nonresidue_is_negative ? FWide::neg(nonresidue_times_im) : nonresidue_times_im;
      return Wide{real_prod + nonresidue_times_im, ab + ba};
    }
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Wide mul_wide(const ComplexExtensionField& xs, const FF& ys)
  {
    return Wide{FF::mul_wide(xs.c0, ys), FF::mul_wide(xs.c1, ys)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Wide mul_wide(const FF& xs, const ComplexExtensionField& ys)
  {
    return mul_wide(ys, xs);
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE ComplexExtensionField reduce(const Wide& xs)
  {
    return ComplexExtensionField{
      FF::template reduce<MODULUS_MULTIPLE>(xs.c0), FF::template reduce<MODULUS_MULTIPLE>(xs.c1)};
  }

  friend HOST_DEVICE_INLINE ComplexExtensionField
  operator*(const ComplexExtensionField& xs, const ComplexExtensionField& ys)
  {
    Wide xy = mul_wide(xs, ys);
    return reduce(xy);
  }

  friend HOST_DEVICE_INLINE ComplexExtensionField operator*(const ComplexExtensionField& xs, const FF& ys)
  {
    Wide xy = mul_wide(xs, ys);
    return reduce(xy);
  }

  friend HOST_DEVICE_INLINE ComplexExtensionField operator*(const FF& ys, const ComplexExtensionField& xs)
  {
    return xs * ys;
  }

  friend HOST_DEVICE_INLINE bool operator==(const ComplexExtensionField& xs, const ComplexExtensionField& ys)
  {
    return (xs.c0 == ys.c0) && (xs.c1 == ys.c1);
  }

  friend HOST_DEVICE_INLINE bool operator!=(const ComplexExtensionField& xs, const ComplexExtensionField& ys)
  {
    return !(xs == ys);
  }

  template <typename Gen, bool IS_3B = false>
  static HOST_DEVICE_INLINE FF mul_weierstrass_b_real(const FF& xs)
  {
    FF r = {};
    constexpr FF b_mult = []() {
      FF b_mult = FF{Gen::weierstrass_b_g2_re};
      if constexpr (!IS_3B) return b_mult;
      typename FF::ff_storage temp = {};
      typename FF::ff_storage modulus = FF::get_modulus();
      host_math::template add_sub_limbs<CONFIG::limbs_count, false, false, true>(
        b_mult.limbs_storage, b_mult.limbs_storage, b_mult.limbs_storage);
      b_mult.limbs_storage =
        host_math::template add_sub_limbs<CONFIG::limbs_count, true, true, true>(b_mult.limbs_storage, modulus, temp)
          ? b_mult.limbs_storage
          : temp;
      host_math::template add_sub_limbs<CONFIG::limbs_count, false, false, true>(
        b_mult.limbs_storage, FF{Gen::weierstrass_b_g2_re}.limbs_storage, b_mult.limbs_storage);
      b_mult.limbs_storage =
        host_math::template add_sub_limbs<CONFIG::limbs_count, true, true, true>(b_mult.limbs_storage, modulus, temp)
          ? b_mult.limbs_storage
          : temp;
      return b_mult;
    }();
    if constexpr (Gen::is_b_u32_g2_re) {
      r = FF::template mul_unsigned<b_mult.limbs_storage.limbs[0], FF>(xs);
      if constexpr (Gen::is_b_neg_g2_re)
        return FF::neg(r);
      else {
        return r;
      }
    } else {
      return b_mult * xs;
    }
  }

  template <typename Gen, bool IS_3B = false>
  static HOST_DEVICE_INLINE FF mul_weierstrass_b_imag(const FF& xs)
  {
    FF r = {};
    constexpr FF b_mult = []() {
      FF b_mult = FF{Gen::weierstrass_b_g2_im};
      if constexpr (!IS_3B) return b_mult;
      typename FF::ff_storage temp = {};
      typename FF::ff_storage modulus = FF::get_modulus();
      host_math::template add_sub_limbs<CONFIG::limbs_count, false, false, true>(
        b_mult.limbs_storage, b_mult.limbs_storage, b_mult.limbs_storage);
      b_mult.limbs_storage =
        host_math::template add_sub_limbs<CONFIG::limbs_count, true, true, true>(b_mult.limbs_storage, modulus, temp)
          ? b_mult.limbs_storage
          : temp;
      host_math::template add_sub_limbs<CONFIG::limbs_count, false, false, true>(
        b_mult.limbs_storage, FF{Gen::weierstrass_b_g2_im}.limbs_storage, b_mult.limbs_storage);
      b_mult.limbs_storage =
        host_math::template add_sub_limbs<CONFIG::limbs_count, true, true, true>(b_mult.limbs_storage, modulus, temp)
          ? b_mult.limbs_storage
          : temp;
      return b_mult;
    }();
    if constexpr (Gen::is_b_u32_g2_im) {
      r = FF::template mul_unsigned<b_mult.limbs_storage.limbs[0], FF>(xs);
      if constexpr (Gen::is_b_neg_g2_im)
        return FF::neg(r);
      else {
        return r;
      }
    } else {
      return b_mult * xs;
    }
  }

  template <typename Gen, bool IS_3B = false>
  static HOST_DEVICE_INLINE ComplexExtensionField mul_weierstrass_b(const ComplexExtensionField& xs)
  {
    const FF xs_real = xs.c0;
    const FF xs_imaginary = xs.c1;
    FF real_prod = mul_weierstrass_b_real<Gen, IS_3B>(xs_real);
    FF imaginary_prod = mul_weierstrass_b_imag<Gen, IS_3B>(xs_imaginary);
    FF re_im = mul_weierstrass_b_real<Gen, IS_3B>(xs_imaginary);
    FF im_re = mul_weierstrass_b_imag<Gen, IS_3B>(xs_real);
    FF nonresidue_times_im = mul_by_nonresidue(imaginary_prod);
    nonresidue_times_im = CONFIG::nonresidue_is_negative ? FF::neg(nonresidue_times_im) : nonresidue_times_im;
    return ComplexExtensionField{real_prod + nonresidue_times_im, re_im + im_re};
  }

  template <const ComplexExtensionField& multiplier>
  static HOST_DEVICE_INLINE ComplexExtensionField mul_const(const ComplexExtensionField& xs)
  {
    static constexpr FF mul_real = multiplier.c0;
    static constexpr FF mul_imaginary = multiplier.c1;
    const FF xs_real = xs.c0;
    const FF xs_imaginary = xs.c1;
    FF real_prod = FF::template mul_const<mul_real>(xs_real);
    FF imaginary_prod = FF::template mul_const<mul_imaginary>(xs_imaginary);
    FF re_im = FF::template mul_const<mul_real>(xs_imaginary);
    FF im_re = FF::template mul_const<mul_imaginary>(xs_real);
    FF nonresidue_times_im = mul_by_nonresidue(imaginary_prod);
    nonresidue_times_im = CONFIG::nonresidue_is_negative ? FF::neg(nonresidue_times_im) : nonresidue_times_im;
    return ComplexExtensionField{real_prod + nonresidue_times_im, re_im + im_re};
  }

  template <uint32_t multiplier, unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE ComplexExtensionField mul_unsigned(const ComplexExtensionField& xs)
  {
    return {FF::template mul_unsigned<multiplier>(xs.c0), FF::template mul_unsigned<multiplier>(xs.c1)};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Wide sqr_wide(const ComplexExtensionField& xs)
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
    return ComplexExtensionField{FF::neg(xs.c0), FF::neg(xs.c1)};
  }

  // inverse of zero is set to be zero which is what we want most of the time
  static constexpr HOST_DEVICE_INLINE ComplexExtensionField inverse(const ComplexExtensionField& xs)
  {
    ComplexExtensionField xs_conjugate = {xs.c0, FF::neg(xs.c1)};
    FF nonresidue_times_im = mul_by_nonresidue(FF::sqr(xs.c1));
    nonresidue_times_im = CONFIG::nonresidue_is_negative ? FF::neg(nonresidue_times_im) : nonresidue_times_im;
    // TODO: wide here
    FF xs_norm_squared = FF::sqr(xs.c0) - nonresidue_times_im;
    return xs_conjugate * ComplexExtensionField{FF::inverse(xs_norm_squared), FF::zero()};
  }

  static constexpr HOST_DEVICE ComplexExtensionField pow(ComplexExtensionField base, int exp)
  {
    ComplexExtensionField res = one();
    while (exp > 0) {
      if (exp & 1) res = res * base;
      base = base * base;
      exp >>= 1;
    }
    return res;
  }

  template <unsigned NLIMBS>
  static constexpr HOST_DEVICE ComplexExtensionField pow(ComplexExtensionField base, storage<NLIMBS> exp)
  {
    ComplexExtensionField res = one();
    while (host_math::is_zero(exp)) {
      if (host_math::get_bit<NLIMBS>(exp, 0)) res = res * base;
      base = base * base;
      exp = host_math::right_shift<NLIMBS, 1>(exp);
    }
    return res;
  }

  /* Receives an array of bytes and its size and returns extension field element. */
  static constexpr HOST_DEVICE_INLINE ComplexExtensionField from(const std::byte* in, unsigned nof_bytes)
  {
    if (nof_bytes < 2 * sizeof(FF)) {
#ifndef __CUDACC__
      ICICLE_LOG_ERROR << "Input size is too small";
#endif // __CUDACC__
      return ComplexExtensionField::zero();
    }
    return ComplexExtensionField{FF::from(in, sizeof(FF)), FF::from(in + sizeof(FF), sizeof(FF))};
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
