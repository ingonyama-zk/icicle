#pragma once
#include <cstdint>

#include "field.cuh"
#include "gpu-utils/modifiers.cuh"
#include "gpu-utils/sharedmem.cuh"

#include "stark_fields/m31.cuh"
#include <cassert>
#include <mutex>

namespace circle_math {

  namespace {
    typedef struct {
      ssize_t a;
      ssize_t b;
      ssize_t c;
    } triple_t;
    triple_t egcd(ssize_t x, ssize_t y) {
      // Initial values for the Bézout coefficients
      ssize_t s1 = 1, t1 = 0;  // Corresponds to (x, 1, 0)
      triple_t result{y, 0, 1};
      ssize_t k, temp;
      
      while (x != 0) {
          k = result.a / x;

          // Update result.a and x as in the Euclidean algorithm
          temp = result.a % x;
          result.a = x;
          x = temp;

          // Update the Bézout coefficients s and t
          temp = result.b - k * s1;
          result.b = s1;
          s1 = temp;

          temp = result.c - k * t1;
          result.c = t1;
          t1 = temp;
      }

      return result;
    }
    
  }

  template <typename CONFIG, class T>
  class CirclePoint
  {
  private:
    friend T;

    typedef typename T::Wide FWide;

    struct PointWide {
      FWide x;
      FWide y;

      friend HOST_DEVICE_INLINE PointWide operator+(PointWide xs, const PointWide& ys)
      {
        return PointWide{xs.x + ys.x, xs.y + ys.y};
      }

      friend HOST_DEVICE_INLINE PointWide operator-(PointWide xs, const PointWide& ys)
      {
        return PointWide{xs.x - ys.x, xs.y - ys.y};
      }
    };

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
      return CirclePoint{{2}, {1268011823}};
    }

    // static constexpr HOST_DEVICE_INLINE CirclePoint to_montgomery(const CirclePoint& xs)
    // {
    //   return CirclePoint{xs.x * FF{CONFIG::montgomery_r}, xs.y * FF{CONFIG::montgomery_r}};
    // }

    // static constexpr HOST_DEVICE_INLINE CirclePoint from_montgomery(const CirclePoint& xs)
    // {
    //   return CirclePoint{xs.x * FF{CONFIG::montgomery_r_inv}, xs.y * FF{CONFIG::montgomery_r_inv}};
    // }

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

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE PointWide
    mul_wide(const CirclePoint& xs, const CirclePoint& ys)
    {
      FWide x_prod = FF::mul_wide(xs.x, ys.x);
      FWide y_prod = FF::mul_wide(xs.y, ys.y);
      FWide prod_of_sums = FF::mul_wide(xs.x + xs.y, ys.x + ys.y);
      FWide nonresidue_times_im = FF::template mul_unsigned<CONFIG::nonresidue>(y_prod);
      nonresidue_times_im = CONFIG::nonresidue_is_negative ? FWide::neg(nonresidue_times_im) : nonresidue_times_im;
      return PointWide{x_prod + nonresidue_times_im, prod_of_sums - x_prod - y_prod};
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE PointWide mul_wide(const CirclePoint& xs, const FF& ys)
    {
      return PointWide{FF::mul_wide(xs.x, ys), FF::mul_wide(xs.y, ys)};
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE PointWide mul_wide(const FF& xs, const CirclePoint& ys)
    {
      return mul_wide(ys, xs);
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE CirclePoint reduce(const PointWide& xs)
    {
      return CirclePoint{
        FF::template reduce<MODULUS_MULTIPLE>(xs.x), FF::template reduce<MODULUS_MULTIPLE>(xs.y)};
    }

    template <class T1, class T2>
    friend HOST_DEVICE_INLINE CirclePoint operator*(const T1& xs, const T2& ys)
    {
      PointWide xy = mul_wide(xs, ys);
      return reduce(xy);
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
    static constexpr HOST_DEVICE_INLINE PointWide sqr_wide(const CirclePoint& xs)
    {
      // TODO: change to a more efficient squaring
      return mul_wide<MODULUS_MULTIPLE>(xs, xs);
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
        if (exp & 1) res = res * base;
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

    // TODO get consts from config
    static HOST_DEVICE_INLINE CirclePoint get_domain_by_index(uint32_t half_coset_initial_index, uint32_t half_coset_step_size, uint32_t domain_size, uint32_t index) {
      uint32_t half_coset_size = domain_size >> 1;
      if (index < half_coset_size) {
        uint64_t global_index = (uint64_t) half_coset_initial_index + (uint64_t) half_coset_step_size * (uint64_t) index;
        return get_point_by_index(global_index & CONFIG::modulus.limbs[0]);
      } else {
        uint64_t global_index = (uint64_t) half_coset_initial_index + (uint64_t) half_coset_step_size * (uint64_t) (index - half_coset_size);
        return get_point_by_index((CONFIG::modulus.limbs[0] + 1 - global_index) & CONFIG::modulus.limbs[0]);
      }
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

  #define M31_CIRCLE_LOG_ORDER 31

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
      assert(log_size <= M31_CIRCLE_LOG_ORDER);
      this->initial_index = initial_index;
      this->log_size = log_size;
      this->step_size = 1 << (M31_CIRCLE_LOG_ORDER - log_size);
      this->initial_point = Point::to_point(initial_index);
      this->step = Point::to_point(step_size);
    }

    static CircleCoset coset_shifted(uint32_t log_shift, uint32_t log_size) {
      return CircleCoset(1 << (M31_CIRCLE_LOG_ORDER - log_size - log_shift), log_size);
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

    size_t size() {
      return 1 << this->log_size;
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

    size_t index_at(size_t index) {
      return this->initial_index + ((this->step_size * index) & (CONFIG::modulus.limbs[0]));
    }

    Point at(size_t index) {
      return Point::to_point(this->index_at(index));
    }

    CircleCoset shift(size_t shift_size) {
      size_t initial_index = this->initial_index + shift_size;
      return CircleCoset{
        initial_index,
        Point::to_point(initial_index),
        this->step_size, 
        this->step, 
        this->log_size
      };
    }

    CircleCoset conjugate() {
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
  };
}