#pragma once
#include <cstdint>

#include "field.cuh"
#include "gpu-utils/modifiers.cuh"
#include "gpu-utils/sharedmem.cuh"

#include "stark_fields/m31.cuh"
#include <cassert>
#include <mutex>

namespace circle_math {

  template <typename CONFIG, class T>
  class CirclePoint
  {
  private:
    friend T;
  public:
    typedef T FF;

    FF x;
    FF y;

    static HOST_DEVICE_INLINE CirclePoint zero()
    {
      return CirclePoint{FF::zero(), FF::zero()};
    }

    static HOST_DEVICE_INLINE CirclePoint one()
    {
      return CirclePoint{FF::one(), FF::zero()};
    }

    static HOST_DEVICE_INLINE CirclePoint generator()
    {
      return CirclePoint{FF{CONFIG::circle_point_generator_x}, FF{CONFIG::circle_point_generator_y}};
    }

    static HOST_INLINE CirclePoint rand_host()
    {
      return CirclePoint{FF::rand_host(), FF::rand_host()};
    }

    static void rand_host_many(CirclePoint* out, int size)
    {
      for (int i = 0; i < size; i++)
        out[i] = rand_host();
    }

    template <unsigned REDUCTION_SIZE = 1>
    static constexpr HOST_DEVICE_INLINE CirclePoint sub_modulus(const CirclePoint& xs)
    {
      return CirclePoint{
        FF::sub_modulus<REDUCTION_SIZE>(&xs.x), FF::sub_modulus<REDUCTION_SIZE>(&xs.y)};
    }

    friend std::ostream& operator<<(std::ostream& os, const CirclePoint& xs)
    {
      os << "{ x: " << xs.x << " }; { y: " << xs.y << " }";
      return os;
    }

    friend HOST_DEVICE_INLINE CirclePoint operator+(CirclePoint xs, const CirclePoint& ys)
    {
      return CirclePoint{xs.x + ys.x, xs.y + ys.y};
    }

    friend HOST_DEVICE_INLINE CirclePoint operator-(CirclePoint xs, const CirclePoint& ys)
    {
      return CirclePoint{xs.x - ys.x, xs.y - ys.y};
    }

    friend HOST_DEVICE_INLINE CirclePoint operator+(FF xs, const CirclePoint& ys)
    {
      return CirclePoint{xs + ys.x, ys.y};
    }

    friend HOST_DEVICE_INLINE CirclePoint operator-(FF xs, const CirclePoint& ys)
    {
      return CirclePoint{xs - ys.x, FF::neg(ys.y)};
    }

    friend HOST_DEVICE_INLINE CirclePoint operator+(CirclePoint xs, const FF& ys)
    {
      return CirclePoint{xs.x + ys, xs.y};
    }

    friend HOST_DEVICE_INLINE CirclePoint operator-(CirclePoint xs, const FF& ys)
    {
      return CirclePoint{xs.x - ys, xs.y};
    }

    friend HOST_DEVICE_INLINE CirclePoint operator*(const CirclePoint& xs, const CirclePoint& ys)
    {
      return CirclePoint{(xs.x * ys.x) - (xs.y * ys.y), (xs.x * ys.y) + (ys.x * xs.y)};
    }

    friend HOST_DEVICE_INLINE bool operator==(const CirclePoint& xs, const CirclePoint& ys)
    {
      return (xs.x == ys.x) && (xs.y == ys.y);
    }

    friend HOST_DEVICE_INLINE bool operator!=(const CirclePoint& xs, const CirclePoint& ys)
    {
      return !(xs == ys);
    }

    template <const CirclePoint& multiplier>
    static HOST_DEVICE_INLINE CirclePoint mul_const(const CirclePoint& xs)
    {
      static constexpr FF mul_x = multiplier.x;
      static constexpr FF mul_y = multiplier.y;
      const FF xs_x = xs.x;
      const FF xs_y = xs.y;
      FF x_prod = FF::template mul_const<mul_x>(xs_x);
      FF y_prod = FF::template mul_const<mul_y>(xs_y);
      FF re_im = FF::template mul_const<mul_x>(xs_y);
      FF im_re = FF::template mul_const<mul_y>(xs_x);
      FF nonresidue_times_im = FF::template mul_unsigned<CONFIG::nonresidue>(y_prod);
      nonresidue_times_im = CONFIG::nonresidue_is_negative ? FF::neg(nonresidue_times_im) : nonresidue_times_im;
      return CirclePoint{x_prod + nonresidue_times_im, re_im + im_re};
    }

    template <uint32_t multiplier, unsigned REDUCTION_SIZE = 1>
    static constexpr HOST_DEVICE_INLINE CirclePoint mul_unsigned(const CirclePoint& xs)
    {
      return {FF::template mul_unsigned<multiplier>(xs.x), FF::template mul_unsigned<multiplier>(xs.y)};
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE CirclePoint sqr(const CirclePoint& xs)
    {
      // TODO: change to a more efficient squaring
      return xs * xs;
    }

    static HOST_DEVICE_INLINE CirclePoint pow(CirclePoint base, int exp)
    {
      CirclePoint res = one();
      while (exp > 0) {
        if (exp & 1) {
          res = res * base;
        }
        base = base * base;
        exp >>= 1;
      }
      return res;
    }

    static HOST_DEVICE_INLINE CirclePoint mul_scalar(CirclePoint base, uint32_t scalar)
    {
      CirclePoint res = zero();
      while (scalar > 0) {
        if (scalar & 1) res = res + base;
        base = base.dbl();
        scalar >>= 1;
      }
      return res;
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE CirclePoint neg(const CirclePoint& xs)
    {
      return CirclePoint{xs.x, FF::neg(xs.y)};
    }

    // inverse of zero is set to be zero which is what we want most of the time
    static HOST_DEVICE_INLINE CirclePoint inverse(const CirclePoint& xs)
    {
      return CirclePoint{xs.x, FF::neg(xs.y)};
    }

    HOST_DEVICE_INLINE CirclePoint dbl() const {
      return *this + *this;
    }

    static HOST_DEVICE_INLINE CirclePoint get_point_by_order(uint32_t order) {
      CirclePoint res = CirclePoint::generator();
      uint32_t l;
  #ifdef __CUDA_ARCH__
      l = __clz(order);
  #else
      l = __builtin_clz(order);
  #endif
      for(size_t i = 0; i < l; ++i)
        res = CirclePoint::sqr(res);
      return res;
    }

    static HOST_DEVICE_INLINE CirclePoint get_point_by_index(uint32_t index) {
      return CirclePoint::pow(CirclePoint::generator(), index);
    }

    static HOST_DEVICE_INLINE CirclePoint to_point(uint32_t index) {
      return CirclePoint::mul_scalar(CirclePoint::generator(), index);
    }

  };
  // template <typename CONFIG, class T>
  // struct SharedMemory<CirclePoint<CONFIG, T>> {
  //   __device__ CirclePoint<CONFIG, T>* getPointer()
  //   {
  //     extern __shared__ CirclePoint<CONFIG, T> s_ext2_scalar_[];
  //     return s_ext2_scalar_;
  //   }
  // };
}