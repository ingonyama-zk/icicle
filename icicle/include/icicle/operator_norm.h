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

namespace opnorm_cpu {
  // We have to use FixedPoint to get reproducible results on other devices
  struct FixedPoint {
    int64_t value;
    static constexpr int64_t scale = 100000000; // 8 decimal places

    HOST_DEVICE static FixedPoint from_double(double f) { return FixedPoint{static_cast<int64_t>(f * scale)}; }

    HOST_DEVICE double to_double() const { return value / static_cast<double>(scale); }

    HOST_DEVICE FixedPoint operator+(const FixedPoint& other) const { return FixedPoint{value + other.value}; }
    HOST_DEVICE FixedPoint operator-(const FixedPoint& other) const { return FixedPoint{value - other.value}; }
    HOST_DEVICE FixedPoint operator*(const FixedPoint& other) const { 
        return FixedPoint{static_cast<int64_t>((__int128_t(value) * other.value) / scale)};
    }
    HOST_DEVICE FixedPoint operator/(const FixedPoint& other) const { return FixedPoint{static_cast<int64_t>((__int128_t(value) * scale) / other.value)}; }

    HOST_DEVICE bool operator>(const FixedPoint& other) const { return value > other.value; }
    HOST_DEVICE bool operator<(const FixedPoint& other) const { return value < other.value; }
    HOST_DEVICE bool operator==(const FixedPoint& other) const { return value == other.value; }
  };

  struct ComplexFixed {
    FixedPoint re, im;

    HOST_DEVICE ComplexFixed operator+(const ComplexFixed& b) const { return ComplexFixed{(re + b.re), (im + b.im)}; }
    HOST_DEVICE ComplexFixed operator-(const ComplexFixed& b) const { return ComplexFixed{(re - b.re), (im - b.im)}; }
    HOST_DEVICE ComplexFixed operator*(const ComplexFixed& b) const {
      FixedPoint real = re * b.re - im * b.im;
      FixedPoint imag = re * b.im + im * b.re;
      return ComplexFixed{real, imag};
    }
    HOST_DEVICE ComplexFixed& operator*=(const ComplexFixed& b) {
      FixedPoint real = re * b.re - im * b.im;
      im = re * b.im + im * b.re;
      re = real;
      return *this;
    }
    HOST_DEVICE FixedPoint abs() const {
      // sqrt is not fixed-point, so convert to double for abs
      // We can implement the fixed point sqrt if needed later
      double sum = re.to_double() * re.to_double() + im.to_double() * im.to_double();
      return FixedPoint::from_double(std::sqrt(sum));
    }
  };

  constexpr size_t N = 64;
  constexpr double PI = 3.14159265358979323846;

  static const ComplexFixed twist[64] = {
    {100000000, 0},
    {99879546, 4906767},
    {99518473, 9801714},
    {98917651, 14673047},
    {98078528, 19509032},
    {97003125, 24298018},
    {95694034, 29028468},
    {94154407, 33688985},
    {92387953, 38268343},
    {90398929, 42755509},
    {88192126, 47139674},
    {85772861, 51410274},
    {83146961, 55557023},
    {80320753, 59569930},
    {77301045, 63439328},
    {74095113, 67155895},
    {70710678, 70710678},
    {67155895, 74095113},
    {63439328, 77301045},
    {59569930, 80320753},
    {55557023, 83146961},
    {51410274, 85772861},
    {47139674, 88192126},
    {42755509, 90398929},
    {38268343, 92387953},
    {33688985, 94154407},
    {29028468, 95694034},
    {24298018, 97003125},
    {19509032, 98078528},
    {14673047, 98917651},
    {9801714, 99518473},
    {4906767, 99879546},
    {0, 100000000},
    {-4906767, 99879546},
    {-9801714, 99518473},
    {-14673047, 98917651},
    {-19509032, 98078528},
    {-24298018, 97003125},
    {-29028468, 95694034},
    {-33688985, 94154407},
    {-38268343, 92387953},
    {-42755509, 90398929},
    {-47139674, 88192126},
    {-51410274, 85772861},
    {-55557023, 83146961},
    {-59569930, 80320753},
    {-63439328, 77301045},
    {-67155895, 74095113},
    {-70710678, 70710678},
    {-74095113, 67155895},
    {-77301045, 63439328},
    {-80320753, 59569930},
    {-83146961, 55557023},
    {-85772861, 51410274},
    {-88192126, 47139674},
    {-90398929, 42755509},
    {-92387953, 38268343},
    {-94154407, 33688985},
    {-95694034, 29028468},
    {-97003125, 24298018},
    {-98078528, 19509032},
    {-98917651, 14673047},
    {-99518473, 9801714},
    {-99879546, 4906767}
  };

  static const ComplexFixed host_wlen_table[6] = {
    { -100000000, 0 }, // len = 2
    { 0, 100000000 }, // len = 4
    { 70710678, 70710678 }, // len = 8
    { 92387953, 38268343 }, // len = 16
    { 98078528, 19509032 }, // len = 32
    { 99518473, 9801714 }, // len = 64
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
      if (inverse) wlen.im = FixedPoint::from_double(-wlen.im.to_double());
      for (size_t i = 0; i < n; i += len) {
        ComplexFixed w = ComplexFixed{FixedPoint::from_double(1.0), FixedPoint::from_double(0)};
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
      FixedPoint scale = FixedPoint::from_double(1.0 / static_cast<double>(n));
      for (ComplexFixed& x : a)
        x = ComplexFixed{x.re * scale, x.im * scale};
    }
  }

  /// @brief Compute the operator norm of a single polynomial: max |FFT(ψᵢ·aᵢ)| over ℂ
  inline int64_t operator_norm(const Poly& a)
  {
    CPoly complex_a;
    for (size_t i = 0; i < N; ++i)
      complex_a[i] = ComplexFixed{FixedPoint::from_double(static_cast<double>(a[i])), FixedPoint{0}} * twist[i];

    fft(complex_a, twist, host_wlen_table);

    FixedPoint max_norm{0};
    for (const auto& x : complex_a) {
      FixedPoint abs_val = x.abs();
      if (abs_val > max_norm) max_norm = abs_val;
    }

    return static_cast<int64_t>(std::ceil(max_norm.to_double()));
  }

} // namespace opnorm_cpu