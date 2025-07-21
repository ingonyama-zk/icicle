
#pragma once

#ifdef __CUDACC__
  #include "gpu-utils/sharedmem.h"
#endif // __CUDACC__
#include "icicle/utils/modifiers.h"
#include <iostream>

template <class BaseField>
class Affine
{
public:
  BaseField x;
  BaseField y;

  static HOST_DEVICE_INLINE Affine zero() { return {BaseField::zero(), BaseField::zero()}; }

  HOST_DEVICE_INLINE Affine neg() const { return {x, y.neg()}; }

  HOST_DEVICE_INLINE Affine to_montgomery() const { return {x.to_montgomery(), y.to_montgomery()}; }

  HOST_DEVICE_INLINE Affine from_montgomery() const { return {x.from_montgomery(), y.from_montgomery()}; }

  HOST_DEVICE_INLINE bool operator==(const Affine& ys) const { return (x == ys.x) && (y == ys.y); }

  HOST_DEVICE_INLINE bool operator!=(const Affine& ys) const { return !(*this == ys); }

  HOST_DEVICE_INLINE bool is_zero() const { return x == BaseField::zero() && y == BaseField::zero(); }

  friend HOST_INLINE std::ostream& operator<<(std::ostream& os, const Affine& point)
  {
    os << "x: " << point.x << "; y: " << point.y;
    return os;
  }
};

#ifdef __CUDACC__
template <class BaseField>
struct SharedMemory<Affine<BaseField>> {
  __device__ Affine<BaseField>* getPointer()
  {
    extern __shared__ Affine<BaseField> s_affine_[];
    return s_affine_;
  }
};
#endif // __CUDACC__
