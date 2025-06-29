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

  struct ComplexFloat {
    float re, im;
    HOST_DEVICE_INLINE ComplexFloat operator+(const ComplexFloat& b) const { return ComplexFloat{re + b.re, im + b.im}; }
    HOST_DEVICE_INLINE ComplexFloat operator-(const ComplexFloat& b) const { return ComplexFloat{re - b.re, im - b.im}; }
    HOST_DEVICE_INLINE ComplexFloat operator*(const ComplexFloat& b) const { return ComplexFloat{re * b.re - im * b.im, re * b.im + im * b.re}; }
    HOST_DEVICE_INLINE ComplexFloat& operator*=(const ComplexFloat& b) { float r = re * b.re - im * b.im; im = re * b.im + im * b.re; re = r; return *this; }
    HOST_DEVICE_INLINE float abs() const { return std::sqrt(re * re + im * im); }
  };

  constexpr size_t N = 64;
  constexpr float PI = 3.14159265358979323846f;

  static const ComplexFloat twist[64] = {
    {1.00000000f, 0.00000000f},
    {0.99879546f, 0.04906767f},
    {0.99518473f, 0.09801714f},
    {0.98917651f, 0.14673047f},
    {0.98078528f, 0.19509032f},
    {0.97003125f, 0.24298018f},
    {0.95694034f, 0.29028468f},
    {0.94154407f, 0.33688985f},
    {0.92387953f, 0.38268343f},
    {0.90398929f, 0.42755509f},
    {0.88192126f, 0.47139674f},
    {0.85772861f, 0.51410274f},
    {0.83146961f, 0.55557023f},
    {0.80320753f, 0.59569930f},
    {0.77301045f, 0.63439328f},
    {0.74095113f, 0.67155895f},
    {0.70710678f, 0.70710678f},
    {0.67155895f, 0.74095113f},
    {0.63439328f, 0.77301045f},
    {0.59569930f, 0.80320753f},
    {0.55557023f, 0.83146961f},
    {0.51410274f, 0.85772861f},
    {0.47139674f, 0.88192126f},
    {0.42755509f, 0.90398929f},
    {0.38268343f, 0.92387953f},
    {0.33688985f, 0.94154407f},
    {0.29028468f, 0.95694034f},
    {0.24298018f, 0.97003125f},
    {0.19509032f, 0.98078528f},
    {0.14673047f, 0.98917651f},
    {0.09801714f, 0.99518473f},
    {0.04906767f, 0.99879546f},
    {0.00000000f, 1.00000000f},
    {-0.04906767f, 0.99879546f},
    {-0.09801714f, 0.99518473f},
    {-0.14673047f, 0.98917651f},
    {-0.19509032f, 0.98078528f},
    {-0.24298018f, 0.97003125f},
    {-0.29028468f, 0.95694034f},
    {-0.33688985f, 0.94154407f},
    {-0.38268343f, 0.92387953f},
    {-0.42755509f, 0.90398929f},
    {-0.47139674f, 0.88192126f},
    {-0.51410274f, 0.85772861f},
    {-0.55557023f, 0.83146961f},
    {-0.59569930f, 0.80320753f},
    {-0.63439328f, 0.77301045f},
    {-0.67155895f, 0.74095113f},
    {-0.70710678f, 0.70710678f},
    {-0.74095113f, 0.67155895f},
    {-0.77301045f, 0.63439328f},
    {-0.80320753f, 0.59569930f},
    {-0.83146961f, 0.55557023f},
    {-0.85772861f, 0.51410274f},
    {-0.88192126f, 0.47139674f},
    {-0.90398929f, 0.42755509f},
    {-0.92387953f, 0.38268343f},
    {-0.94154407f, 0.33688985f},
    {-0.95694034f, 0.29028468f},
    {-0.97003125f, 0.24298018f},
    {-0.98078528f, 0.19509032f},
    {-0.98917651f, 0.14673047f},
    {-0.99518473f, 0.09801714f},
    {-0.99879546f, 0.04906767f}
  };

  using Poly = std::array<int64_t, N>;
  using CPoly = std::array<ComplexFloat, N>;

  template <const uint64_t Q>
  HOST_DEVICE_INLINE int64_t balance(int64_t x)
  {
    return x >= (Q / 2) ? x - Q : x;
  }

  /// @brief In-place Cooley-Tukey radix-2 FFT (float32)
  HOST_DEVICE_INLINE void fft(CPoly& a, const ComplexFloat* twist, bool inverse = false)
  {
    const size_t n = a.size();

    // Bit-reversal permutation
    for (size_t i = 1, j = 0; i < n; ++i) {
      size_t bit = n >> 1;
      for (; j & bit; bit >>= 1)
        j ^= bit;
      j ^= bit;
      if (i < j) {
        ComplexFloat tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
      }
    }

    for (size_t len = 2; len <= n; len <<= 1) {
      float angle = 2.0f * PI / static_cast<float>(len) * (inverse ? -1.0f : 1.0f);
      ComplexFloat wlen{cosf(angle), sinf(angle)};
      for (size_t i = 0; i < n; i += len) {
        ComplexFloat w{1.0f, 0.0f};
        for (size_t j = 0; j < len / 2; ++j) {
          ComplexFloat u = a[i + j];
          ComplexFloat v = a[i + j + len / 2] * w;
          a[i + j] = u + v;
          a[i + j + len / 2] = u - v;
          w *= wlen;
        }
      }
    }

    if (inverse) {
      float scale = 1.0f / static_cast<float>(n);
      for (ComplexFloat& x : a)
        x = ComplexFloat{x.re * scale, x.im * scale};
    }
  }

  /// @brief Compute the operator norm of a single polynomial: max |FFT(ψᵢ·aᵢ)| over ℂ
  inline int64_t operator_norm(const Poly& a)
  {
    CPoly complex_a;
    for (size_t i = 0; i < N; ++i)
      complex_a[i] = ComplexFloat{static_cast<float>(a[i]), 0.0f} * twist[i];

    fft(complex_a, twist);

    float max_norm = 0.0f;
    for (const auto& x : complex_a) {
      max_norm = std::max(max_norm, x.abs());
    }

    float precision = 1e-3f;
    float rounded = static_cast<int>(max_norm * 100) / 100.0f;
    // printf("cpu max norm: %f rounded: %f\n", max_norm, rounded);
    return static_cast<int64_t>(ceilf(rounded));
  }

} // namespace opnorm_cpu