#pragma once

#include "icicle/math/modular_arithmetic.h"

template <class CONFIG>
class IntegerRing : public ModArith<IntegerRing<CONFIG>, CONFIG>
{
  // By deriving from ModArith<IntegerRing> (CRTP) we get operands defined for the type IntegerRing

public:
  static constexpr HOST_DEVICE bool has_inverse(const IntegerRing& xs)
  {
    // Note: inverse returns zero when no inverse
    auto xs_inv = IntegerRing::inverse(xs);
    return xs_inv != IntegerRing::zero();
  }
};

template <class CONFIG>
struct std::hash<IntegerRing<CONFIG>> {
  std::size_t operator()(const IntegerRing<CONFIG>& key) const
  {
    std::size_t hash = 0;
    // boost hashing, see
    // https://stackoverflow.com/questions/35985960/c-why-is-boosthash-combine-the-best-way-to-combine-hash-values/35991300#35991300
    for (int i = 0; i < CONFIG::limbs_count; i++)
      hash ^= std::hash<uint32_t>()(key.limbs_storage.limbs[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};

#ifdef __CUDACC__
template <class CONFIG>
struct SharedMemory<IntegerRing<CONFIG>> {
  __device__ IntegerRing<CONFIG>* getPointer()
  {
    extern __shared__ IntegerRing<CONFIG> s_scalar_[];
    return s_scalar_;
  }
};

#endif // __CUDACC__
