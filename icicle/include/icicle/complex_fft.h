#pragma once

/// @file complex_fft.h
/// @brief Header file for complex FFT operations.
/// This is actually negacyclic-fft of size N=64 for the PolyRing Zq[X]/(X^N + 1),

// TODO: move this to cpu backend

#include <array>
#include <complex>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace negacyclic_fft_cpu {

  constexpr size_t N = 64;
  constexpr double PI = 3.14159265358979323846;
  using Complex = std::complex<double>;
  using Poly = std::array<uint64_t, N>;
  using CPoly = std::array<Complex, N>;

  // Return psi^i (pre or post-twist), where psi^2 = root of unity
  constexpr std::array<Complex, N> compute_twist(bool inverse = false)
  {
    std::array<Complex, N> twist{};
    double angle_unit = PI / N; // 2π/2N = π/N
    for (size_t i = 0; i < N; ++i) {
      double angle = angle_unit * i;
      if (inverse) angle = -angle;
      twist[i] = Complex(std::cos(angle), std::sin(angle));
    }
    return twist;
  }

  // In-place Cooley-Tukey FFT (radix-2)
  void fft(CPoly& a, bool inverse = false)
  {
    size_t n = a.size();
    // Bit reversal
    for (size_t i = 1, j = 0; i < n; ++i) {
      size_t bit = n >> 1;
      for (; j & bit; bit >>= 1)
        j ^= bit;
      j ^= bit;
      if (i < j) std::swap(a[i], a[j]);
    }

    for (size_t len = 2; len <= n; len <<= 1) {
      double ang = 2 * PI / len * (inverse ? -1 : 1);
      Complex wlen(std::cos(ang), std::sin(ang));
      for (size_t i = 0; i < n; i += len) {
        Complex w = 1;
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
      for (Complex& x : a)
        x /= n;
    }
  }

  // Entry point: compute operator norm of polynomial a in Zq[X]/(X⁶⁴ + 1)
  double operator_norm(const Poly& a, uint64_t q)
  {
    static const auto twist = compute_twist();
    static const auto twist_inv = compute_twist(true);

    // Step 1: Convert to Complex, apply pre-twist
    CPoly complex_a;
    for (size_t i = 0; i < N; ++i)
      complex_a[i] = Complex(static_cast<double>(a[i] % q), 0.0) * twist[i];

    // Step 2: Forward FFT
    fft(complex_a);

    // Step 3: Post-twist
    for (size_t i = 0; i < N; ++i)
      complex_a[i] *= twist_inv[i];

    // Step 4: Operator norm = max_i |value_i|
    double max_norm = 0.0;
    for (const auto& x : complex_a)
      max_norm = std::max(max_norm, std::abs(x));

    return max_norm;
  }
} // namespace negacyclic_fft_cpu