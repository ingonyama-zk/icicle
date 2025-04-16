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

    HOST_DEVICE_INLINE Wide operator+(const Wide& ys) const { 
      return Wide{c0 + ys.c0, c1 + ys.c1}; 
    }

    HOST_DEVICE_INLINE Wide operator-(const Wide& ys) const { 
      return Wide{c0 - ys.c0, c1 - ys.c1}; 
    }

    constexpr HOST_DEVICE_INLINE Wide neg() const { return Wide{c0.neg(), c1.neg()}; }

    constexpr HOST_DEVICE_INLINE ComplexExtensionField reduce() const
    {
      return ComplexExtensionField{c0.reduce(), c1.reduce()};
    }
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

  static constexpr HOST_DEVICE_INLINE ComplexExtensionField from(uint32_t val)
  {
    return ComplexExtensionField{FF::from(val), FF::zero()};
  }

  /* Receives an array of bytes and its size and returns extension field element. */
  static constexpr HOST_DEVICE_INLINE ComplexExtensionField from(const std::byte* in, unsigned nof_bytes)
  {
    if (nof_bytes < 2 * sizeof(FF)) { ICICLE_LOG_ERROR << "Input size is too small"; }
    return ComplexExtensionField{FF::from(in, sizeof(FF)), FF::from(in + sizeof(FF), sizeof(FF))};
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

  constexpr HOST_DEVICE_INLINE ComplexExtensionField to_montgomery() const
  {
    return ComplexExtensionField{c0.to_montgomery(), c1.to_montgomery()};
  }

  constexpr HOST_DEVICE_INLINE ComplexExtensionField from_montgomery() const
  {
    return ComplexExtensionField{c0.from_montgomery(), c1.from_montgomery()};
  }

  template <unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE ComplexExtensionField sub_modulus(const ComplexExtensionField& xs)
  {
    return ComplexExtensionField{FF::sub_modulus<REDUCTION_SIZE>(xs.c0), FF::sub_modulus<REDUCTION_SIZE>(xs.c1)};
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
    return ComplexExtensionField{xs - ys.c0, ys.c1.neg()};
  }

  friend HOST_DEVICE_INLINE ComplexExtensionField operator+(ComplexExtensionField xs, const FF& ys)
  {
    return ComplexExtensionField{xs.c0 + ys, xs.c1};
  }

  friend HOST_DEVICE_INLINE ComplexExtensionField operator-(ComplexExtensionField xs, const FF& ys)
  {
    return ComplexExtensionField{xs.c0 - ys, xs.c1};
  }

  constexpr HOST_DEVICE_INLINE ComplexExtensionField operator-() const { return this->neg(); }

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
      return xs.reduce().mul_wide(CONFIG::nonresidue);
    }
  }

  constexpr HOST_DEVICE_INLINE Wide mul_wide(const ComplexExtensionField& ys) const
  {
    FWide real_prod = c0.mul_wide(ys.c0);
    FWide imaginary_prod = c1.mul_wide(ys.c1);
    FWide prod_of_sums = (c0 + c1).mul_wide(ys.c0 + ys.c1);
    FWide nonresidue_times_im = mul_by_nonresidue(imaginary_prod);
    nonresidue_times_im = CONFIG::nonresidue_is_negative ? nonresidue_times_im.neg() : nonresidue_times_im;
    return Wide{real_prod + nonresidue_times_im, prod_of_sums - real_prod - imaginary_prod};
  }

  constexpr HOST_DEVICE_INLINE Wide mul_wide(const FF& ys) const { return Wide{c0.mul_wide(ys), c1.mul_wide(ys)}; }

  static constexpr HOST_DEVICE_INLINE Wide mul_wide(const FF& xs, const ComplexExtensionField& ys)
  {
    return ys.mul_wide(xs);
  }

  template <class T2>
  friend HOST_DEVICE_INLINE ComplexExtensionField operator*(const ComplexExtensionField& xs, const T2& ys)
  {
    Wide xy = xs.mul_wide(ys);
    return xy.reduce();
  }

  HOST_DEVICE_INLINE bool operator==(const ComplexExtensionField& ys) const
  {
    return (c0 == ys.c0) && (c1 == ys.c1);
  }

  HOST_DEVICE_INLINE bool operator!=(const ComplexExtensionField& ys) const
  {
    return !(*this == ys);
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
        return r.neg();
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
        return r.neg();
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
    nonresidue_times_im = CONFIG::nonresidue_is_negative ? nonresidue_times_im.neg() : nonresidue_times_im;
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
    nonresidue_times_im = CONFIG::nonresidue_is_negative ? nonresidue_times_im.neg() : nonresidue_times_im;
    return ComplexExtensionField{real_prod + nonresidue_times_im, re_im + im_re};
  }

  template <uint32_t multiplier>
  static constexpr HOST_DEVICE_INLINE ComplexExtensionField mul_unsigned(const ComplexExtensionField& xs)
  {
    return {FF::template mul_unsigned<multiplier>(xs.c0), FF::template mul_unsigned<multiplier>(xs.c1)};
  }

  constexpr HOST_DEVICE_INLINE Wide sqr_wide() const
  {
    // TODO: change to a more efficient squaring
    return mul_wide(*this, *this);
  }

  constexpr HOST_DEVICE_INLINE ComplexExtensionField sqr() const
  {
    // TODO: change to a more efficient squaring
    return *this * *this;
  }

  constexpr HOST_DEVICE_INLINE ComplexExtensionField neg() const { return ComplexExtensionField{c0.neg(), c1.neg()}; }

  // inverse of zero is set to be zero which is what we want most of the time
  constexpr HOST_DEVICE_INLINE ComplexExtensionField inverse() const
  {
    ComplexExtensionField xs_conjugate = {c0, c1.neg()};
    FF nonresidue_times_im = mul_by_nonresidue(c1.sqr());
    nonresidue_times_im = CONFIG::nonresidue_is_negative ? nonresidue_times_im.neg() : nonresidue_times_im;
    // TODO: wide here
    FF xs_norm_squared = c0.sqr() - nonresidue_times_im;
    return xs_conjugate * ComplexExtensionField{xs_norm_squared.inverse(), FF::zero()};
  }

  constexpr HOST_DEVICE ComplexExtensionField pow(int exp) const
  {
    ComplexExtensionField res = one();
    ComplexExtensionField base = *this;
    while (exp > 0) {
      if (exp & 1) res = res * base;
      base = base.sqr();
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
