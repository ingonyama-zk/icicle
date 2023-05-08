#pragma once

#include "field.cuh"

#define HOST_INLINE __host__ __forceinline__
#define DEVICE_INLINE __device__ __forceinline__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__

template <class CONFIG> class ExtensionField {
  private:
    struct ExtensionWide {
      Field<CONFIG>::Wide real_wide;
      Field<CONFIG>::Wide imaginary_wide;
      
      ExtensionField HOST_DEVICE_INLINE get_lower() {
        return ExtensionField { real_wide.get_lower(), imaginary_wide.get_lower() };
      }

      ExtensionField HOST_DEVICE_INLINE get_higher_with_slack() {
        return ExtensionField { real_wide.get_higher_with_slack(), imaginary_wide.get_higher_with_slack() };
      }
    };

    friend HOST_DEVICE_INLINE ExtensionWide operator+(ExtensionWide xs, const ExtensionWide& ys) {   
      return ExtensionField { xs.real + ys.real, xs.imaginary + ys.imaginary };
    }

    // an incomplete impl that assumes that xs > ys
    friend HOST_DEVICE_INLINE ExtensionWide operator-(ExtensionWide xs, const ExtensionWide& ys) {   
      return ExtensionField { xs.real - ys.real, xs.imaginary - ys.imaginary };
    }

  public:
    Field<CONFIG> real;
    Field<CONFIG> imaginary;

    static constexpr HOST_DEVICE_INLINE ExtensionField zero() {
      return ExtensionField { CONFIG::zero, CONFIG::zero };
    }

    static constexpr HOST_DEVICE_INLINE ExtensionField one() {
      return ExtensionField { CONFIG::one, CONFIG::zero };
    }

    static HOST_INLINE ExtensionField rand_host() {
      return ExtensionField { Field<CONFIG>::rand_host(), Field<CONFIG>::rand_host() };
    }

    template <unsigned REDUCTION_SIZE = 1> static constexpr HOST_DEVICE_INLINE ExtensionField reduce(const ExtensionField &xs) {
      return ExtensionField { Field<CONFIG>::reduce<REDUCTION_SIZE>(&xs.real), Field<CONFIG>::reduce<REDUCTION_SIZE>(&xs.imaginary) };
    }

    friend std::ostream& operator<<(std::ostream& os, const ExtensionField& xs) {
      os << "{ Real: " << xs.real << " }; { Imaginary: " << xs.imaginary << " }";
      return os;
    }

    friend HOST_DEVICE_INLINE ExtensionField operator+(ExtensionField xs, const ExtensionField& ys) {
      return ExtensionField { xs.real + ys.real, xs.imaginary + ys.imaginary };
    }

    friend HOST_DEVICE_INLINE ExtensionField operator-(ExtensionField xs, const ExtensionField& ys) {
      return ExtensionField { xs.real - ys.real, xs.imaginary - ys.imaginary };
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE wide mul_wide(const Field& xs, const Field& ys) {
      wide rs = {};
      multiply_raw(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
      return rs;
    }

    friend HOST_DEVICE_INLINE Field operator*(const Field& xs, const Field& ys) {
      wide xy = mul_wide(xs, ys);
      Field xy_hi = xy.get_higher_with_slack();
      wide l = {};
      multiply_raw(xy_hi.limbs_storage, get_m(), l.limbs_storage);
      Field l_hi = l.get_higher_with_slack();
      wide lp = {};
      multiply_raw(l_hi.limbs_storage, get_modulus(), lp.limbs_storage);
      wide r_wide = xy - lp;
      wide r_wide_reduced = {};
      uint32_t reduced = sub_limbs<true>(r_wide.limbs_storage, modulus_wide(), r_wide_reduced.limbs_storage);
      r_wide = reduced ? r_wide : r_wide_reduced;
      Field r = r_wide.get_lower();
      return reduce<1>(r);
    }

    friend HOST_DEVICE_INLINE bool operator==(const Field& xs, const Field& ys) {
    #ifdef __CUDA_ARCH__
      const uint32_t *x = xs.limbs_storage.limbs;
      const uint32_t *y = ys.limbs_storage.limbs;
      uint32_t limbs_or = x[0] ^ y[0];
  #pragma unroll
      for (unsigned i = 1; i < TLC; i++)
        limbs_or |= x[i] ^ y[i];
      return limbs_or == 0;
    #else
      for (unsigned i = 0; i < TLC; i++)
      if (xs.limbs_storage.limbs[i] != ys.limbs_storage.limbs[i])
        return false;
      return true;
    #endif
    }

    friend HOST_DEVICE_INLINE bool operator!=(const Field& xs, const Field& ys) {
      return !(xs == ys);
    }

    template <unsigned REDUCTION_SIZE = 1>
    static constexpr HOST_DEVICE_INLINE Field mul(const unsigned scalar, const Field &xs) {
      Field rs = {};
      Field temp = xs;
      unsigned l = scalar;
      bool is_zero = true;
  #ifdef __CUDA_ARCH__
  #pragma unroll
  #endif
      for (unsigned i = 0; i < 32; i++) {
        if (l & 1) {
          rs = is_zero ? temp : (rs + temp);
          is_zero = false;
        }
        l >>= 1;
        if (l == 0)
          break;
        temp = temp + temp;
      }
      return rs;
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE wide sqr_wide(const Field& xs) {
      // TODO: change to a more efficient squaring
      return mul_wide<MODULUS_MULTIPLE>(xs, xs);
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE Field sqr(const Field& xs) {
      // TODO: change to a more efficient squaring
      return xs * xs;
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE Field neg(const Field& xs) {
      const ff_storage modulus = get_modulus<MODULUS_MULTIPLE>();
      Field rs = {};
      sub_limbs<false>(modulus, xs.limbs_storage, rs.limbs_storage);
      return rs;
    }

    template <unsigned MODULUS_MULTIPLE = 1> 
    static constexpr HOST_DEVICE_INLINE Field div2(const Field &xs) {
      const uint32_t *x = xs.limbs_storage.limbs;
      Field rs = {};
      uint32_t *r = rs.limbs_storage.limbs;
  #ifdef __CUDA_ARCH__
  #pragma unroll
  #endif
      for (unsigned i = 0; i < TLC - 1; i++) {
    #ifdef __CUDA_ARCH__
        r[i] = __funnelshift_rc(x[i], x[i + 1], 1);
    #else
        r[i] = (x[i] >> 1) | (x[i + 1] << 31);
    #endif
      }
      r[TLC - 1] = x[TLC - 1] >> 1;
      return reduce<MODULUS_MULTIPLE>(rs);
    }

    static constexpr HOST_DEVICE_INLINE bool lt(const Field &xs, const Field &ys) {
      ff_storage dummy = {};
      uint32_t carry = sub_limbs<true>(xs.limbs_storage, ys.limbs_storage, dummy);
      return carry;
    }

    static constexpr HOST_DEVICE_INLINE bool is_odd(const Field &xs) { 
      return xs.limbs_storage.limbs[0] & 1;
    }

    static constexpr HOST_DEVICE_INLINE bool is_even(const Field &xs) { 
      return ~xs.limbs_storage.limbs[0] & 1;
    }

    // inverse assumes that xs is nonzero
    static constexpr HOST_DEVICE_INLINE Field inverse(const Field& xs) {
      constexpr Field one = Field { CONFIG::one };
      constexpr ff_storage modulus = CONFIG::modulus;
      Field u = xs;
      Field v = Field { modulus };
      Field b = one;
      Field c = {};
      while (!(u == one) && !(v == one)) {
        while (is_even(u)) {
          u = div2(u);
          if (is_odd(b))
            add_limbs<false>(b.limbs_storage, modulus, b.limbs_storage);
          b = div2(b);
        }
        while (is_even(v)) {
          v = div2(v);
          if (is_odd(c))
            add_limbs<false>(c.limbs_storage, modulus, c.limbs_storage);
          c = div2(c);
        }
        if (lt(v, u)) {
          u = u - v;
          b = b - c;
        } else {
          v = v - u;
          c = c - b;
        }
      }
      return (u == one) ?  b : c;
    }
};
