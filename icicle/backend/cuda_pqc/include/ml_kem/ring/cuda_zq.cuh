#pragma once

#include <stdint.h>

namespace icicle::pqc::ml_kem {

  class Zq
  {
  public:
    static constexpr uint16_t q = 3329;

    // Default constructor initializes to zero
    __forceinline__ __host__ __device__ constexpr Zq() : value(0) {}

    // External input constructor — reduces via modulo
    __forceinline__ __host__ __device__ explicit Zq(uint32_t v) : value(static_cast<uint16_t>(v % q)) {}

    // Constexpr constructor for compile-time values
    __forceinline__ __host__ __device__ constexpr Zq(int v) : value(static_cast<uint16_t>(v % q)) {}

    // Internal: use when value is already in [0, q)
    __forceinline__ __host__ __device__ static Zq from_raw(uint32_t reduced_value)
    {
      Zq z;
      z.value = static_cast<uint16_t>(reduced_value);
      return z;
    }

    // Arithmetic operators
    __forceinline__ __host__ __device__ Zq operator+(const Zq& other) const
    {
      uint16_t r = value + other.value;
      r = (r >= q) ? r - q : r;
      return from_raw(r);
    }

    __forceinline__ __host__ __device__ Zq operator-(const Zq& other) const
    {
      uint16_t r = (value >= other.value) ? value - other.value : value + q - other.value;
      return from_raw(r);
    }

    __forceinline__ __host__ __device__ Zq operator*(const Zq& other) const
    {
      uint16_t r = ((uint32_t)value * other.value) % q;
      return from_raw(r);
    }

    __forceinline__ __host__ __device__ Zq& operator+=(const Zq& other)
    {
      *this = *this + other;
      return *this;
    }

    __forceinline__ __host__ __device__ Zq& operator*=(const Zq& other)
    {
      *this = *this * other;
      return *this;
    }

    // Comparison
    __forceinline__ __host__ __device__ bool operator==(const Zq& other) const { return value == other.value; }
    __forceinline__ __host__ __device__ bool operator!=(const Zq& other) const { return value != other.value; }

    // Access raw underlying value
    __forceinline__ __host__ __device__ uint16_t raw() const { return value; }

    // Returns value in symmetric range: [-q/2, q/2]
    __forceinline__ __host__ __device__ int16_t centered() const
    {
      int16_t c = static_cast<int16_t>(value);
      return (c > static_cast<int16_t>(q / 2)) ? c - q : c;
    }

    // Rounds to bit: 0 or 1 depending on threshold
    __forceinline__ __host__ __device__ int round() const { return (value > q / 2) ? 1 : 0; }

  private:
    uint16_t value; // always in [0, q)
  };

} // namespace icicle::pqc::ml_kem
