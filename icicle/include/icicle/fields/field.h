#pragma once

#include "icicle/math/modular_arithmetic.h"

template <class CONFIG>
class Field : public ModArith<Field<CONFIG>, CONFIG>
{
  // By deriving from ModArith<Field> (CRTP) we get operands defined for the type Field

public:
  static constexpr unsigned TLC = CONFIG::limbs_count;
  static constexpr unsigned NBITS = CONFIG::modulus_bit_count;
  typedef storage<TLC> ff_storage;

  template <typename Gen, bool IS_3B = false>
  static HOST_DEVICE_INLINE Field mul_weierstrass_b(const Field& xs)
  {
    Field r = {};
    constexpr Field b_mult = []() {
      Field b_mult = Field{Gen::weierstrass_b};
      if constexpr (!IS_3B) return b_mult;
      ff_storage temp = {};
      ff_storage modulus = ModArith<Field<CONFIG>, CONFIG>::template get_modulus<>();
      host_math::template add_sub_limbs<TLC, false, false, true>(
        b_mult.limbs_storage, b_mult.limbs_storage, b_mult.limbs_storage);
      b_mult.limbs_storage =
        host_math::template add_sub_limbs<TLC, true, true, true>(b_mult.limbs_storage, modulus, temp)
          ? b_mult.limbs_storage
          : temp;
      host_math::template add_sub_limbs<TLC, false, false, true>(
        b_mult.limbs_storage, Field{Gen::weierstrass_b}.limbs_storage, b_mult.limbs_storage);
      b_mult.limbs_storage =
        host_math::template add_sub_limbs<TLC, true, true, true>(b_mult.limbs_storage, modulus, temp)
          ? b_mult.limbs_storage
          : temp;
      return b_mult;
    }();

    if constexpr (Gen::is_b_u32) { // assumes that 3b is also u32
      r = ModArith<Field<CONFIG>, CONFIG>::template mul_unsigned<b_mult.limbs_storage.limbs[0], Field>(xs);
      if constexpr (Gen::is_b_neg)
        return ModArith<Field<CONFIG>, CONFIG>::template neg<b_mult.limbs_storage.limbs[0], Field>(r);
      else {
        return r;
      }
    } else {
      return b_mult * xs;
    }
  }
};

template <class CONFIG>
struct std::hash<Field<CONFIG>> {
  std::size_t operator()(const Field<CONFIG>& key) const
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
struct SharedMemory<Field<CONFIG>> {
  __device__ Field<CONFIG>* getPointer()
  {
    extern __shared__ Field<CONFIG> s_scalar_[];
    return s_scalar_;
  }
};

#endif // __CUDACC__
