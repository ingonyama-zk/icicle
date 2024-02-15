#pragma once

#include "field.cuh"

template <class FF>
class Affine
{
public:
  FF x;
  FF y;

  static HOST_DEVICE_INLINE Affine neg(const Affine& point) { return {point.x, FF::neg(point.y)}; }

  static HOST_DEVICE_INLINE Affine zero() { return {FF::zero(), FF::zero()}; }

  static HOST_DEVICE_INLINE Affine ToMontgomery(const Affine& point)
  {
    return {FF::ToMontgomery(point.x), FF::ToMontgomery(point.y)};
  }

  static HOST_DEVICE_INLINE Affine FromMontgomery(const Affine& point)
  {
    return {FF::FromMontgomery(point.x), FF::FromMontgomery(point.y)};
  }

  friend HOST_DEVICE_INLINE bool operator==(const Affine& xs, const Affine& ys)
  {
    return (xs.x == ys.x) && (xs.y == ys.y);
  }

  friend HOST_INLINE std::ostream& operator<<(std::ostream& os, const Affine& point)
  {
    os << "x: " << point.x << "; y: " << point.y;
    return os;
  }
};
