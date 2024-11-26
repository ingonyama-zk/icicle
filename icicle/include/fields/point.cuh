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

  template <typename CONFIG, class T>
  class CircleCoset {
  typedef CirclePoint<CONFIG, T> Point;
  private:
    CircleCoset(size_t index, Point point, size_t step_size, Point step_point, size_t log_size)
          : initial_index(index), initial_point(point), step_size(step_size), step(step_point), log_size(log_size) {}
  public:
    size_t initial_index;
    Point initial_point;
    size_t step_size;
    Point step;
    size_t log_size;

    CircleCoset(size_t initial_index, uint32_t log_size) {
      assert(log_size <= CONFIG::modulus_bit_count);
      this->initial_index = initial_index;
      this->log_size = log_size;
      this->step_size = 1 << (CONFIG::modulus_bit_count - log_size);
      this->initial_point = Point::to_point(initial_index);
      this->step = Point::to_point(step_size);
    }

    static CircleCoset coset_shifted(uint32_t log_shift, uint32_t log_size) {
      return CircleCoset(1 << (CONFIG::modulus_bit_count - log_size - log_shift), log_size);
    }

    // Creates a coset of the form <G_n>.
    static CircleCoset subgroup(uint32_t log_size) {
      return CircleCoset(0, log_size);
    }

    // Creates a coset of the form G_2n + <G_n>.
    static CircleCoset odds(uint32_t log_size) {
      return coset_shifted(1, log_size); // log 2
    }

    // Creates a coset of the form G_4n + <G_n>.
    static CircleCoset half_odds(uint32_t log_size) {
      return coset_shifted(2, log_size); // log 4
    }

    size_t lg_size() const {
      return this->log_size;
    }

    size_t size() const {
      return 1 << this->lg_size();
    }

    CircleCoset dbl() const {
      assert(this->log_size);
      return CircleCoset{
        this->initial_index * 2, 
        this->initial_point.dbl(), 
        this->step_size * 2, 
        this->step.dbl(), 
        this->log_size? this->log_size - 1 : 0 };
    }

    size_t index_at(size_t index) const {
      return this->initial_index + ((this->step_size * index) & (CONFIG::modulus.limbs[0]));
    }

    Point at(size_t index) const {
      return Point::to_point(this->index_at(index));
    }

    CircleCoset shift(size_t shift_size) const {
      size_t initial_index = this->initial_index + shift_size;
      return CircleCoset{
        initial_index,
        Point::to_point(initial_index),
        this->step_size, 
        this->step, 
        this->log_size
      };
    }

    CircleCoset conjugate() const {
      size_t initial_index = ((CONFIG::modulus.limbs[0] + 1) - this->initial_index) & (CONFIG::modulus.limbs[0]);
      size_t step_size = ((CONFIG::modulus.limbs[0] + 1) - this->step_size) & (CONFIG::modulus.limbs[0]);
      return CircleCoset{
        initial_index,
        Point::to_point(initial_index),
        step_size,
        Point::to_point(step_size),
        this->log_size
      };
    }

    template <typename C, typename U>
    friend std::ostream& operator<<(std::ostream& os, const CircleCoset<C, U>& coset) {
        os << "CircleCoset { "
          << "initial_index: " << coset.initial_index << ", "
          << "initial_point: " << coset.initial_point << ", "
          << "step_size: " << coset.step_size << ", "
          << "step: " << coset.step << ", "
          << "log_size: " << coset.log_size
          << " }";
        return os;
    }
  };
  template <typename CONFIG, class T>
  class CircleDomain {
  public:
    typedef CirclePoint<CONFIG, T> Point;
    CircleCoset<CONFIG, T> coset;
    // Constructors
    // CircleDomain(size_t initial_index, uint32_t log_size)
    //     : coset(initial_index, log_size) {}
    CircleDomain<CONFIG, T>(const CircleCoset<CONFIG, T>& coset) : coset(coset) {}
    CircleDomain<CONFIG, T>(uint32_t log_size) : coset(CircleCoset<CONFIG, T>::half_odds(log_size - 1)) {}

    // Override log_size method
    size_t lg_size() const {
      return coset.lg_size() + 1;
    }

    // Forward other methods to coset
    size_t size() const {
      return 1 << this->lg_size();
    }

    size_t index_at(size_t index) const {
      return coset.index_at(index);
    }

    Point at(size_t index) const {
      return coset.at(index);
    }

    CircleDomain shift(size_t shift_size) const {
      return CircleDomain(coset.shift(shift_size));
    }

    CircleDomain split(size_t log_parts, size_t* shifts) const {
      CircleDomain<CONFIG, T> subdomain = CircleDomain<CONFIG, T>(CircleCoset<CONFIG, T>(coset.initial_index, coset.log_size - log_parts));
      for (size_t i = 0; i < (1 << log_parts); ++i) {
        shifts[i] = coset.step_size * i;
      }
      return subdomain;
    }
  };
}