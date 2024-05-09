#pragma once

#include "../gpu-utils/sharedmem.cuh"
#include "../gpu-utils/modifiers.cuh"
#include <iostream>

template <class FF>
class Affine
{
public:
  FF x;
  FF y;

  static Affine neg(const Affine& point) { return {point.x, FF::neg(point.y)}; }

  static Affine zero() { return {FF::zero(), FF::zero()}; }

  static Affine to_montgomery(const Affine& point)
  {
    return {FF::to_montgomery(point.x), FF::to_montgomery(point.y)};
  }

  static Affine from_montgomery(const Affine& point)
  {
    return {FF::from_montgomery(point.x), FF::from_montgomery(point.y)};
  }

  friend bool operator==(const Affine& xs, const Affine& ys)
  {
    return (xs.x == ys.x) && (xs.y == ys.y);
  }

  friend std::ostream& operator<<(std::ostream& os, const Affine& point)
  {
    os << "x: " << point.x << "; y: " << point.y;
    return os;
  }
};

template <class FF>
struct SharedMemory<Affine<FF>> {
  Affine<FF>* getPointer()
  {
    Affine<FF> *s_affine_ = nullptr;
    return s_affine_;
  }
};
