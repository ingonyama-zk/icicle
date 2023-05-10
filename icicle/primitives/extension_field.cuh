#pragma once

#include "field.cuh"

#define HOST_INLINE __host__ __forceinline__
#define DEVICE_INLINE __device__ __forceinline__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__

template <typename CONFIG> class ExtensionField {
  private:
    typedef typename Field<CONFIG>::Wide FWide;

    struct ExtensionWide {
      FWide real_wide;
      FWide imaginary_wide;
      
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
    typedef Field<CONFIG> FF;

    FF real;
    FF imaginary;

    static constexpr HOST_DEVICE_INLINE ExtensionField zero() {
      return ExtensionField { FF::zero(), FF::zero() };
    }

    static constexpr HOST_DEVICE_INLINE ExtensionField one() {
      return ExtensionField { FF::one(), FF::zero() };
    }

    static constexpr HOST_DEVICE_INLINE ExtensionField generator_x() {
      #ifndef __CUDA_ARCH__
      std::cout << "in extension generator";
      #endif
      return ExtensionField { FF { CONFIG::generator_x_re }, FF { CONFIG::generator_x_im } };
    }

    static constexpr HOST_DEVICE_INLINE ExtensionField generator_y() {
      return ExtensionField { FF { CONFIG::generator_y_re }, FF { CONFIG::generator_y_im } };
    }


    static HOST_INLINE ExtensionField rand_host() {
      return ExtensionField { FF::rand_host(), FF::rand_host() };
    }

    template <unsigned REDUCTION_SIZE = 1> static constexpr HOST_DEVICE_INLINE ExtensionField reduce(const ExtensionField &xs) {
      return ExtensionField { FF::reduce<REDUCTION_SIZE>(&xs.real), FF::reduce<REDUCTION_SIZE>(&xs.imaginary) };
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
    static constexpr HOST_DEVICE_INLINE ExtensionWide mul_wide(const ExtensionField& xs, const ExtensionField& ys) {
      FWide real_prod = FF::mul_wide(xs.real * ys.real);
      FWide imaginary_prod = FF::mul_wide(xs.imaginary * ys.imaginary);
      FWide prod_of_sums = FF::mul_wide(xs.real + xs.imaginary, ys.real + ys.imaginary);
      FWide i_sq_times_im = FF::mul(CONFIG::i_squared, imaginary_prod);
      i_sq_times_im = CONFIG::i_squared_is_negative ? FF::neg(i_sq_times_im) : i_sq_times_im;
      return ExtensionField { real_prod + i_sq_times_im, prod_of_sums - real_prod - imaginary_prod };
    }

    friend HOST_DEVICE_INLINE ExtensionField operator*(const ExtensionField& xs, const ExtensionField& ys) {
      FF real_prod = xs.real * ys.real;
      FF imaginary_prod = xs.imaginary * ys.imaginary;
      FF prod_of_sums = (xs.real + xs.imaginary) * (ys.real + ys.imaginary);
      FF i_sq_times_im = FF::mul(CONFIG::i_squared, imaginary_prod);
      i_sq_times_im = CONFIG::i_squared_is_negative ? FF::neg(i_sq_times_im) : i_sq_times_im;
      return ExtensionField { real_prod + i_sq_times_im, prod_of_sums - real_prod - imaginary_prod };
    }

    friend HOST_DEVICE_INLINE bool operator==(const ExtensionField& xs, const ExtensionField& ys) {
      return (xs.real == ys.real) && (xs.imaginary == ys.imaginary);
    }

    friend HOST_DEVICE_INLINE bool operator!=(const ExtensionField& xs, const ExtensionField& ys) {
      #ifndef __CUDA_ARCH__
      std::cout << "X Re: " << xs.real << std::endl;
      std::cout << "X Im: " << xs.imaginary << std::endl;
      std::cout << "Y Re: " << ys.real << std::endl;
      std::cout << "Y Im: " << ys.imaginary << std::endl;
      #endif
      return !(xs == ys);
    }

    template <class T, unsigned REDUCTION_SIZE = 1>
    static constexpr HOST_DEVICE_INLINE T mul(const unsigned scalar, const T &xs) {
      T rs = {};
      T temp = xs;
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
        // todo: impl doubling
        temp = temp + temp;
      }
      return rs;
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE ExtensionWide sqr_wide(const ExtensionField& xs) {
      // TODO: change to a more efficient squaring
      return mul_wide<MODULUS_MULTIPLE>(xs, xs);
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE ExtensionField sqr(const ExtensionField& xs) {
      // TODO: change to a more efficient squaring
      return xs * xs;
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE ExtensionField neg(const ExtensionField& xs) {
      return ExtensionField { FF::neg(xs.real), FF::neg(xs.imaginary) };
    }

    // inverse assumes that xs is nonzero
    static constexpr HOST_DEVICE_INLINE ExtensionField inverse(const ExtensionField& xs) {
      ExtensionField xs_conjugate = { xs.real, FF::neg(xs.imaginary) };
      // TODO: wide here
      FF xs_modulus_squared = FF::sqr(xs.real) + FF::sqr(xs.imaginary);
      return xs_conjugate * FF::inverse(xs_modulus_squared);
    }
};
