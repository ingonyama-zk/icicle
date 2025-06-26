#pragma once

/// @file operator_norm.h
/// @brief Operator norm estimation for polynomials in Zq[X]/(X^N + 1) using float32 FFT.
/// This version is tailored for single-threaded (or per-thread) use, compatible with GPU kernels.
///
/// Notes:
/// - N = 64 (fixed), over polynomials modulo X^N + 1
/// - Input is uint64_t, assumed reduced mod q
/// - Uses float32 complex FFT with twist/untwist to compute the spectral ℓ∞ norm
/// - No batching or heap allocations; ready for device-side integration

#include <array>
#include <complex>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace negacyclic_fft_cpu {

  constexpr size_t N = 64;
  constexpr float PI = 3.14159265358979323846f;

  using Complex = std::complex<float>;
  using Poly = std::array<uint64_t, N>;
  using CPoly = std::array<Complex, N>;

  /// @brief Compute psi^i where psi = exp(pi * i / N), used for twist/untwist
  inline std::array<Complex, N> compute_twist(bool inverse = false)
  {
    std::array<Complex, N> twist{};
    float angle_unit = PI / static_cast<float>(N);
    for (size_t i = 0; i < N; ++i) {
      float angle = angle_unit * static_cast<float>(i);
      if (inverse) angle = -angle;
      twist[i] = Complex(std::cos(angle), std::sin(angle));
    }
    return twist;
  }

  /// @brief In-place Cooley-Tukey radix-2 FFT (float32)
  inline void fft(CPoly& a, bool inverse = false)
  {
    const size_t n = a.size();

    // Bit-reversal permutation
    for (size_t i = 1, j = 0; i < n; ++i) {
      size_t bit = n >> 1;
      for (; j & bit; bit >>= 1)
        j ^= bit;
      j ^= bit;
      if (i < j) std::swap(a[i], a[j]);
    }

    for (size_t len = 2; len <= n; len <<= 1) {
      float angle = 2.0f * PI / static_cast<float>(len) * (inverse ? -1.0f : 1.0f);
      Complex wlen(std::cos(angle), std::sin(angle));
      for (size_t i = 0; i < n; i += len) {
        Complex w = 1.0f;
        for (size_t j = 0; j < len / 2; ++j) {
          Complex u = a[i + j];
          Complex v = a[i + j + len / 2] * w;
          a[i + j] = u + v;
          a[i + j + len / 2] = u - v;
          w *= wlen;
        }
      }
    }

    if (inverse) {
      float scale = 1.0f / static_cast<float>(n);
      for (Complex& x : a)
        x *= scale;
    }
  }

  /// @brief Compute the operator norm of a single polynomial: max |FFT(ψᵢ·aᵢ)| over ℂ
  inline uint64_t operator_norm(const Poly& a)
  {
    static const auto twist = compute_twist(false);
    static const auto twist_inv = compute_twist(true);

    CPoly complex_a;
    for (size_t i = 0; i < N; ++i)
      complex_a[i] = Complex(static_cast<float>(a[i]), 0.0f) * twist[i];

    fft(complex_a);

    for (size_t i = 0; i < N; ++i)
      complex_a[i] *= twist_inv[i];

    float max_norm = 0.0f;
    for (const auto& x : complex_a)
      max_norm = std::max(max_norm, std::abs(x));

    return static_cast<uint64_t>(std::ceil(max_norm));
  }

} // namespace negacyclic_fft_cpu