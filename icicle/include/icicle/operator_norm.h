#pragma once

/// @file operator_norm.h
/// @brief Operator norm estimation for polynomials in Zq[X]/(X^N + 1) using float32 FFT.
/// This version is tailored for single-threaded (or per-thread) use, compatible with GPU kernels.
///
/// Notes:
/// - N = 64 (fixed), over polynomials modulo X^N + 1
/// - Input is int64_t, assumed reduced mod q
/// - Uses float32 complex FFT with twist to compute the spectral ℓ∞ norm
/// - No batching or heap allocations; ready for device-side integration

#include <array>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cassert>

namespace opnorm {
  // We have to use FixedPoint to get reproducible results on other devices
  struct FixedPoint {
    int32_t value;
    static constexpr float scale = 1000000.0f; // 6 decimal places

    HOST_DEVICE static int32_t reduce(int64_t num, int64_t denom)
    {
      int32_t result;
      if (num >= 0)
        result = static_cast<int32_t>((num + (denom / 2)) / denom);
      else
        result = static_cast<int32_t>((num - (denom / 2)) / denom);
      return result;
    }

    // This only works for f in [-2147, 2147]
    HOST_DEVICE static FixedPoint from_int32_t(int32_t f) { return FixedPoint{f * static_cast<int32_t>(scale)}; }
    // This only works for f in [-2147.0f, 2147.0f]
    HOST_DEVICE static FixedPoint from_float(float f) { return FixedPoint{static_cast<int32_t>(f * scale)}; }
    HOST_DEVICE float to_float() const { return value / scale; }

    HOST_DEVICE FixedPoint operator+(const FixedPoint& other) const { return FixedPoint{value + other.value}; }
    HOST_DEVICE FixedPoint operator-(const FixedPoint& other) const { return FixedPoint{value - other.value}; }
    HOST_DEVICE FixedPoint operator-() const { return FixedPoint{-value}; }
    HOST_DEVICE FixedPoint operator*(const FixedPoint& other) const
    {
      int64_t prod = static_cast<int64_t>(value) * other.value;
      return FixedPoint{reduce(prod, static_cast<int64_t>(scale))};
    }
    HOST_DEVICE FixedPoint operator/(const FixedPoint& other) const
    {
      int64_t num = static_cast<int64_t>(value) * static_cast<int64_t>(scale);
      return FixedPoint{reduce(num, static_cast<int64_t>(other.value))};
    }

    HOST_DEVICE bool operator>(const FixedPoint& other) const { return value > other.value; }
    HOST_DEVICE bool operator<(const FixedPoint& other) const { return value < other.value; }
    HOST_DEVICE bool operator==(const FixedPoint& other) const { return value == other.value; }
  };

  struct ComplexFixed {
    FixedPoint re, im;

    HOST_DEVICE ComplexFixed operator+(const ComplexFixed& b) const { return ComplexFixed{(re + b.re), (im + b.im)}; }
    HOST_DEVICE ComplexFixed operator-(const ComplexFixed& b) const { return ComplexFixed{(re - b.re), (im - b.im)}; }
    HOST_DEVICE ComplexFixed operator*(const ComplexFixed& b) const
    {
      FixedPoint real = re * b.re - im * b.im;
      FixedPoint imag = re * b.im + im * b.re;
      return ComplexFixed{real, imag};
    }
    HOST_DEVICE ComplexFixed& operator*=(const ComplexFixed& b)
    {
      FixedPoint real = re * b.re - im * b.im;
      im = re * b.im + im * b.re;
      re = real;
      return *this;
    }
    HOST_DEVICE float abs() const
    {
      float sum = re.to_float() * re.to_float() + im.to_float() * im.to_float();
      return sqrtf(sum);
    }
  };

  constexpr size_t N = 64;
  constexpr double PI = 3.14159265358979323846;

  constexpr static const ComplexFixed twist[64] = {
    {1000000, 0},      {998795, 49067},   {995184, 98017},   {989176, 146730},  {980785, 195090},  {970031, 242980},
    {956940, 290284},  {941544, 336889},  {923879, 382683},  {903989, 427555},  {881921, 471396},  {857728, 514102},
    {831469, 555570},  {803207, 595699},  {773010, 634393},  {740951, 671558},  {707106, 707106},  {671558, 740951},
    {634393, 773010},  {595699, 803207},  {555570, 831469},  {514102, 857728},  {471396, 881921},  {427555, 903989},
    {382683, 923879},  {336889, 941544},  {290284, 956940},  {242980, 970031},  {195090, 980785},  {146730, 989176},
    {98017, 995184},   {49067, 998795},   {0, 1000000},      {-49067, 998795},  {-98017, 995184},  {-146730, 989176},
    {-195090, 980785}, {-242980, 970031}, {-290284, 956940}, {-336889, 941544}, {-382683, 923879}, {-427555, 903989},
    {-471396, 881921}, {-514102, 857728}, {-555570, 831469}, {-595699, 803207}, {-634393, 773010}, {-671558, 740951},
    {-707106, 707106}, {-740951, 671558}, {-773010, 634393}, {-803207, 595699}, {-831469, 555570}, {-857728, 514102},
    {-881921, 471396}, {-903989, 427555}, {-923879, 382683}, {-941544, 336889}, {-956940, 290284}, {-970031, 242980},
    {-980785, 195090}, {-989176, 146730}, {-995184, 98017},  {-998795, 49067}};

  constexpr static const ComplexFixed host_wlen_table[6] = {
    {-1000000, 0},    // len = 2
    {0, 1000000},     // len = 4
    {707106, 707106}, // len = 8
    {923879, 382683}, // len = 16
    {980785, 195090}, // len = 32
    {995184, 98017},  // len = 64
  };

  using Poly = std::array<int64_t, N>;
  using CPoly = std::array<ComplexFixed, N>;

  template <const uint64_t Q>
  HOST_DEVICE_INLINE int64_t balance(int64_t x)
  {
    return x >= (Q / 2) ? x - Q : x;
  }

  /// @brief In-place Cooley-Tukey radix-2 FFT (float32)
  HOST_DEVICE_INLINE void fft(CPoly& a, const ComplexFixed* twist, const ComplexFixed* wlen_table, bool inverse = false)
  {
    const size_t n = a.size();

    // Bit-reversal permutation
    for (size_t i = 1, j = 0; i < n; ++i) {
      size_t bit = n >> 1;
      for (; j & bit; bit >>= 1)
        j ^= bit;
      j ^= bit;
      if (i < j) {
        ComplexFixed tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
      }
    }

    for (size_t len = 2, stage = 0; len <= n; ++stage, len <<= 1) {
      ComplexFixed wlen = wlen_table[stage];
      if (inverse) wlen.im = -wlen.im;
      for (size_t i = 0; i < n; i += len) {
        ComplexFixed w = ComplexFixed{FixedPoint::from_float(1.0f), FixedPoint::from_float(0.0f)};
        for (size_t j = 0; j < len / 2; ++j) {
          ComplexFixed u = a[i + j];
          ComplexFixed v = a[i + j + len / 2] * w;
          a[i + j] = u + v;
          a[i + j + len / 2] = u - v;
          w *= wlen;
        }
      }
    }

    if (inverse) {
      FixedPoint scale = FixedPoint::from_float(1.0f / static_cast<float>(n));
      for (ComplexFixed& x : a)
        x = ComplexFixed{x.re * scale, x.im * scale};
    }
  }

  /// @brief Compute the operator norm of a single polynomial: max |FFT(ψᵢ·aᵢ)| over ℂ
  inline int64_t operator_norm(const Poly& a)
  {
    CPoly complex_a;
    for (size_t i = 0; i < N; ++i) {
      complex_a[i] = ComplexFixed{FixedPoint::from_int32_t(a[i]), FixedPoint{0}} * twist[i];
    }

    fft(complex_a, twist, host_wlen_table);

    float max_norm = 0.0f;
    for (const auto& x : complex_a) {
      float abs_val = x.abs();
      if (abs_val > max_norm) max_norm = abs_val;
    }
    // Since we are interested in low norm polynomials, we need to address for floating point imprecision
    max_norm += 0.000001f;
    return static_cast<int64_t>(ceilf(max_norm));
  }

} // namespace opnorm