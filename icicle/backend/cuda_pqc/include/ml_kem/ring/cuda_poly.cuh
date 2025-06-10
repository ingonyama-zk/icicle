#pragma once

#include <stdint.h>

namespace icicle::pqc::ml_kem {
  template <uint32_t N, typename T>
  struct PolyView {
  private:
    T* _data;

  public:
    __forceinline__ __host__ __device__ explicit PolyView(T* ptr) : _data(ptr) {}

    __forceinline__ __host__ __device__ T* data() { return _data; }

    __forceinline__ __host__ __device__ const T* data() const { return _data; }

    __forceinline__ __host__ __device__ T& operator[](uint32_t i) { return _data[i]; }

    __forceinline__ __host__ __device__ const T& operator[](uint32_t i) const { return _data[i]; }

    __forceinline__ __host__ __device__ static constexpr uint32_t byte_size() { return N * sizeof(T); }
  };

  template <uint32_t N, uint8_t K, typename T>
  struct PolyVecView {
  private:
    T* _data;

  public:
    __forceinline__ __host__ __device__ explicit PolyVecView(T* ptr) : _data(ptr) {}

    __forceinline__ __host__ __device__ T* data() { return _data; }

    __forceinline__ __host__ __device__ const T* data() const { return _data; }

    __forceinline__ __host__ __device__ PolyView<N, T> operator[](uint8_t i)
    {
      return PolyView<N, T>(_data + (uint32_t)i * N);
    }

    __forceinline__ __host__ __device__ const PolyView<N, T> operator[](uint8_t i) const
    {
      return PolyView<N, T>(_data + (uint32_t)i * N);
    }

    __forceinline__ __host__ __device__ T* get_raw_element(uint8_t i) { return _data + (uint32_t)i * N; }

    __forceinline__ __host__ __device__ const T* get_raw_element(uint8_t i) const { return _data + (uint32_t)i * N; }

    __forceinline__ __host__ __device__ static constexpr uint32_t byte_size() { return N * K * sizeof(T); }
  };

  template <uint32_t N, uint8_t COLS, uint8_t ROWS, typename T>
  struct PolyMatrixView {
  private:
    T* _data;

  public:
    __forceinline__ __host__ __device__ explicit PolyMatrixView(T* ptr) : _data(ptr) {}

    __forceinline__ __host__ __device__ T* data() { return _data; }

    __forceinline__ __host__ __device__ const T* data() const { return _data; }

    // row r (0 ≤ r < ROWS) as a PolyVecView<N,COLS,T>
    __forceinline__ __host__ __device__ PolyVecView<N, COLS, T> operator[](uint8_t r)
    {
      return PolyVecView<N, COLS, T>(_data + (uint32_t)r * COLS * N);
    }

    __forceinline__ __host__ __device__ const PolyVecView<N, COLS, T> operator[](uint8_t r) const
    {
      return PolyVecView<N, COLS, T>(_data + (uint32_t)r * COLS * N);
    }

    // single-poly access at (r,c)
    __forceinline__ __host__ __device__ PolyView<N, T> operator()(uint8_t r, uint8_t c)
    {
      return PolyView<N, T>(_data + ((uint32_t)r * COLS + c) * N);
    }

    __forceinline__ __host__ __device__ const PolyView<N, T> operator()(uint8_t r, uint8_t c) const
    {
      return PolyView<N, T>(_data + ((uint32_t)r * COLS + c) * N);
    }

    __forceinline__ __host__ __device__ T* get_raw_element(uint8_t r, uint8_t c)
    {
      return _data + ((uint32_t)r * COLS + c) * N;
    }

    __forceinline__ __host__ __device__ const T* get_raw_element(uint8_t r, uint8_t c) const
    {
      return _data + ((uint32_t)r * COLS + c) * N;
    }

    __forceinline__ __host__ __device__ static constexpr uint32_t byte_size() { return ROWS * COLS * N * sizeof(T); }
  };
} // namespace icicle::pqc::ml_kem