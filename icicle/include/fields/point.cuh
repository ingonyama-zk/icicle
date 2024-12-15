#pragma once
#include <cstdint>

#include "field.cuh"
#include "gpu-utils/modifiers.cuh"
#include "gpu-utils/sharedmem.cuh"

#include "stark_fields/m31.cuh"
#include <cassert>
#include <mutex>
#include <unordered_map>
#include <functional>

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
      return CirclePoint{xs.x * ys.x - xs.y * ys.y, xs.x * ys.y + xs.y * ys.x};
    }

    friend HOST_DEVICE_INLINE CirclePoint operator-(CirclePoint xs, const CirclePoint& ys)
    {
      return xs + CirclePoint::neg(ys);
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

    friend HOST_DEVICE_INLINE bool operator==(const CirclePoint& xs, const CirclePoint& ys)
    {
      return (xs.x == ys.x) && (xs.y == ys.y);
    }

    friend HOST_DEVICE_INLINE bool operator!=(const CirclePoint& xs, const CirclePoint& ys)
    {
      return !(xs == ys);
    }

    static HOST_DEVICE_INLINE CirclePoint mul_scalar(CirclePoint base, uint32_t scalar)
    {
      CirclePoint res = zero();
      CirclePoint cur = base;
      while (scalar > 0) {
        if (scalar & 1) {
          res = res + cur;
        }
        cur = cur.dbl();
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

    static HOST_DEVICE_INLINE FF dbl_x(FF x) {
      x = FF::sqr(x);
      return x + x - FF::one();
    }

    static HOST_DEVICE_INLINE CirclePoint index_to_point(uint32_t index) {
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
    HOST_DEVICE_INLINE CircleCoset(uint64_t index, Point point, uint64_t step_size, Point step_point, uint64_t log_size)
          : initial_index(index), initial_point(point), step_size(step_size), step(step_point), log_size(log_size) {}
  public:
    uint64_t initial_index;
    Point initial_point;
    uint64_t step_size;
    Point step;
    uint32_t log_size;

    HOST_DEVICE_INLINE CircleCoset(uint64_t initial_index, uint32_t log_size) {
      assert(log_size <= CONFIG::modulus_bit_count);
      this->initial_index = initial_index;
      this->log_size = log_size;
      this->step_size = 1 << (CONFIG::modulus_bit_count - log_size);
      this->initial_point = Point::index_to_point(initial_index);
      this->step = Point::index_to_point(step_size);
    }

    static HOST_DEVICE_INLINE CircleCoset coset_shifted(uint32_t log_shift, uint32_t log_size) {
      return CircleCoset(1 << (CONFIG::modulus_bit_count - log_size - log_shift), log_size);
    }

    // Creates a coset of the form <G_n>.
    static HOST_DEVICE_INLINE CircleCoset subgroup(uint32_t log_size) {
      return CircleCoset(0, log_size);
    }

    // Creates a coset of the form G_2n + <G_n>.
    static HOST_DEVICE_INLINE CircleCoset odds(uint32_t log_size) {
      return coset_shifted(1, log_size); // log 2
    }

    // Creates a coset of the form G_4n + <G_n>.
    static HOST_DEVICE_INLINE CircleCoset half_odds(uint32_t log_size) {
      return coset_shifted(2, log_size); // log 4
    }

    HOST_DEVICE_INLINE uint32_t lg_size() const {
      return this->log_size;
    }

    HOST_DEVICE_INLINE uint64_t size() const {
      return 1 << this->lg_size();
    }

    HOST_DEVICE_INLINE CircleCoset dbl() const {
      assert(this->log_size);
      return CircleCoset{
        this->initial_index * 2, 
        this->initial_point.dbl(), 
        this->step_size * 2, 
        this->step.dbl(), 
        this->log_size? this->log_size - 1 : 0 };
    }

    HOST_DEVICE_INLINE uint64_t index_at(uint64_t index) const {
      return this->initial_index + ((this->step_size * index) & (CONFIG::modulus.limbs[0]));
    }

    HOST_DEVICE_INLINE Point at(uint64_t index) const {
      return Point::index_to_point(this->index_at(index));
    }

    friend HOST_DEVICE_INLINE bool operator==(const CircleCoset& xs, const CircleCoset& ys)
    {
      return (xs.initial_index == ys.initial_index) && (xs.log_size == ys.log_size);
    }

    HOST_DEVICE_INLINE CircleCoset shift(uint64_t shift_size) const {
      uint64_t initial_index = this->initial_index + shift_size;
      return CircleCoset{
        initial_index,
        Point::index_to_point(initial_index),
        this->step_size, 
        this->step, 
        this->log_size
      };
    }

    HOST_DEVICE_INLINE CircleCoset conjugate() const {
      uint64_t initial_index = ((CONFIG::modulus.limbs[0] + 1) - this->initial_index) & (CONFIG::modulus.limbs[0]);
      uint64_t step_size = ((CONFIG::modulus.limbs[0] + 1) - this->step_size) & (CONFIG::modulus.limbs[0]);
      return CircleCoset{
        initial_index,
        Point::index_to_point(initial_index),
        step_size,
        Point::index_to_point(step_size),
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
  struct CircleCosetHash {
    std::size_t operator()(const CircleCoset<CONFIG, T>& coset) const {
      std::size_t h1 = std::hash<uint64_t>{}(coset.initial_index);
      std::size_t h2 = std::hash<uint32_t>{}(coset.log_size);
      return h1 ^ (h2 << 1);
    }
  };

  DEVICE_INLINE uint64_t bit_reverse_index(uint64_t index, uint32_t log_size) {
    if (log_size == 0) {
      return index;
    }
    return __brevll(index) >> ((sizeof(uint64_t) << 3) - log_size);
  }

  template <typename D, typename T>
  __global__ void compute_domain_twiddles(D domain, size_t size, T *twiddles_inversed_reversed_index) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
      twiddles_inversed_reversed_index[idx] = T::inverse(domain.at(bit_reverse_index(idx << 1, domain.lg_size())).y);
    }
  }

  template <typename CONFIG, class T>
  class CircleDomain {
  private:
    static HOST_INLINE void compute_twiddles(const CircleDomain<CONFIG, T>& domain) {
      if (!CircleDomain<CONFIG, T>::twiddles_inversed_reversed_index.count(domain.coset)) {
        auto size = domain.coset.size();
        T *d_twiddles;
        cudaMalloc(&d_twiddles, size);
        int block_dim = size < 512 ? size : 512; 
        int num_blocks = block_dim < 512 ? 1 : (size + block_dim - 1) / block_dim;
        compute_domain_twiddles<CircleDomain<CONFIG, T>, T><<<num_blocks, block_dim>>>(domain, size, d_twiddles);
        CircleDomain<CONFIG, T>::twiddles_inversed_reversed_index[domain.coset] = d_twiddles;
      }
    }
  public:
    typedef CirclePoint<CONFIG, T> Point;
    CircleCoset<CONFIG, T> coset;
    static std::unordered_map<CircleCoset<CONFIG, T>, T*, CircleCosetHash<CONFIG, T>> twiddles_inversed_reversed_index;
    CircleDomain<CONFIG, T>(const CircleCoset<CONFIG, T>& coset) : coset(coset) {
      compute_twiddles(*this);
    }
    CircleDomain<CONFIG, T>(uint32_t log_size) : coset(CircleCoset<CONFIG, T>::half_odds(log_size - 1)) {
      compute_twiddles(*this);
    }

    // Override log_size method
    HOST_DEVICE_INLINE uint32_t lg_size() const {
      return coset.lg_size() + 1;
    }

    // Forward other methods to coset
    HOST_DEVICE_INLINE uint64_t size() const {
      return 1 << this->lg_size();
    }

    HOST_DEVICE_INLINE uint64_t index_at(uint64_t index) const {
      if (index < coset.size()) {
        return coset.index_at(index);
      } else {
        // assume field log size <= 32
        return (((uint32_t)1 << CONFIG::modulus_bit_count) - coset.index_at(index - coset.size())); // reduce is not neccesary
      }
    }

    HOST_DEVICE_INLINE Point at(uint64_t index) const {
      return Point::index_to_point(this->index_at(index));
    }

    HOST_DEVICE_INLINE CircleDomain shift(uint64_t shift_size) const {
      return CircleDomain(coset.shift(shift_size));
    }

    HOST_DEVICE_INLINE CircleDomain split(uint64_t log_parts, uint64_t* shifts) const {
      CircleDomain<CONFIG, T> subdomain = CircleDomain<CONFIG, T>(CircleCoset<CONFIG, T>(coset.initial_index, coset.log_size - log_parts));
      for (uint64_t i = 0; i < (1 << log_parts); ++i) {
        shifts[i] = coset.step_size * i;
      }
      return subdomain;
    }

    HOST_INLINE void get_twiddles(T **twiddles) const {
      *twiddles = CircleDomain<CONFIG, T>::twiddles_inversed_reversed_index.at(this->coset);
    }
  };

  template <typename CONFIG, class T>
  std::unordered_map<CircleCoset<CONFIG, T>, T*, CircleCosetHash<CONFIG, T>> CircleDomain<CONFIG, T>::twiddles_inversed_reversed_index;

  template <typename CONFIG, class T>
  class LineDomain {
  private:
    static HOST_DEVICE_INLINE uint32_t log_order(T x) {
      uint32_t result = 0;
      T one = T::one();
      while (x != one) {
        x = Point::dbl_x(x);
        ++result;
      }
      return result;
    }
  public:
    typedef CirclePoint<CONFIG, T> Point;
    CircleCoset<CONFIG, T> coset;
    HOST_DEVICE_INLINE LineDomain<CONFIG, T>(const CircleCoset<CONFIG, T>& coset) : coset(coset) {
      uint64_t size = coset.size();
      if (size > 2) {
        assert(LineDomain::log_order(coset.initial_point.x) >= LineDomain::log_order(coset.step.x) + 2);
      }
      if (size == 2) {
        assert(coset.initial_point.x != T::zero());
      }
    }

    HOST_DEVICE_INLINE LineDomain<CONFIG, T>(uint64_t initial_index, uint32_t log_size) : coset(CircleCoset<CONFIG, T>(initial_index, log_size)) {
      uint64_t size = coset.size();
      if (size > 2) {
        assert(LineDomain::log_order(coset.initial_point.x) >= LineDomain::log_order(coset.step.x) + 2);
      }
      if (size == 2) {
        assert(coset.initial_point.x != T::zero());
      }
    }

    HOST_DEVICE_INLINE uint32_t lg_size() const {
      return coset.lg_size();
    }

    HOST_DEVICE_INLINE T at(uint64_t index) const {
      return coset.at(index).x;
    }

    HOST_DEVICE_INLINE uint64_t size() const {
      return coset.size();
    }

    HOST_DEVICE_INLINE LineDomain dbl() const {
      return LineDomain(coset.dbl());
    }
  };
}