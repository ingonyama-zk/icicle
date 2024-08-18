#pragma GCC optimize ("no-strict-aliasing")

#pragma once

#include <cstdint>
#include "icicle/utils/modifiers.h"
#include "icicle/fields/storage.h"
namespace host_math {

  // return x + y with T operands
  template <typename T>
  static constexpr __host__ T add(const T x, const T y)
  {
    return x + y;
  }

  // return x + y + carry with T operands
  template <typename T>
  static constexpr __host__ T addc(const T x, const T y, const T carry)
  {
    return x + y + carry;
  }

  // return x + y and carry out with T operands
  template <typename T>
  static constexpr __host__ T add_cc(const T x, const T y, T& carry)
  {
    T result = x + y;
    carry = x > result;
    return result;
  }

  // return x + y + carry and carry out  with T operands
  template <typename T>
  static constexpr __host__ T addc_cc(const T x, const T y, T& carry)
  {
    const T result = x + y + carry;
    carry = carry && x >= result || !carry && x > result;
    return result;
  }

  // return x - y with T operands
  template <typename T>
  static constexpr __host__ T sub(const T x, const T y)
  {
    return x - y;
  }

  //    return x - y - borrow with T operands
  template <typename T>
  static constexpr __host__ T subc(const T x, const T y, const T borrow)
  {
    return x - y - borrow;
  }

  //    return x - y and borrow out with T operands
  template <typename T>
  static constexpr __host__ T sub_cc(const T x, const T y, T& borrow)
  {
    T result = x - y;
    borrow = x < result;
    return result;
  }

  //    return x - y - borrow and borrow out with T operands
  template <typename T>
  static constexpr __host__ T subc_cc(const T x, const T y, T& borrow)
  {
    const T result = x - y - borrow;
    borrow = borrow && x <= result || !borrow && x < result;
    return result;
  }

  // return x * y + z + carry and carry out with uint32_t operands
  static constexpr __host__ uint32_t madc_cc(const uint32_t x, const uint32_t y, const uint32_t z, uint32_t& carry)
  {
    uint64_t r = static_cast<uint64_t>(x) * y + z + carry;
    carry = (uint32_t)(r >> 32);
    uint32_t result = r & 0xffffffff;
    return result;
  }

  static constexpr __host__ uint64_t madc_cc_64(const uint64_t x, const uint64_t y, const uint64_t z, uint64_t& carry)
  {
    __uint128_t r = static_cast<__uint128_t>(x) * y + z + carry;
    carry = (uint64_t)(r >> 64);
    uint64_t result = r & 0xffffffffffffffff;
    return result;
  }

  template <unsigned OPS_COUNT = UINT32_MAX, bool CARRY_IN = false, bool CARRY_OUT = false>
  struct carry_chain {
    unsigned index;

    constexpr HOST_INLINE carry_chain() : index(0) {}

    template <typename T>
    constexpr HOST_INLINE T add(const T x, const T y, T& carry)
    {
      index++;
      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
        return host_math::add(x, y);
      else if (index == 1 && !CARRY_IN)
        return host_math::add_cc(x, y, carry);
      else if (index < OPS_COUNT || CARRY_OUT)
        return host_math::addc_cc(x, y, carry);
      else
        return host_math::addc(x, y, carry);
    }

    template <typename T>
    constexpr HOST_INLINE T sub(const T x, const T y, T& carry)
    {
      index++;
      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
        return host_math::sub(x, y);
      else if (index == 1 && !CARRY_IN)
        return host_math::sub_cc(x, y, carry);
      else if (index < OPS_COUNT || CARRY_OUT)
        return host_math::subc_cc(x, y, carry);
      else
        return host_math::subc(x, y, carry);
    }
  };

  template <unsigned NLIMBS, bool SUBTRACT, bool CARRY_OUT>
  static constexpr HOST_INLINE uint32_t add_sub_limbs_32(const uint32_t* x, const uint32_t* y, uint32_t* r)
  {
    uint32_t carry = 0;
    carry_chain<NLIMBS, false, CARRY_OUT> chain;
    for (unsigned i = 0; i < NLIMBS; i++)
      r[i] = SUBTRACT ? chain.sub(x[i], y[i], carry) : chain.add(x[i], y[i], carry);
    return CARRY_OUT ? carry : 0;
  }

  template <unsigned NLIMBS, bool SUBTRACT, bool CARRY_OUT>
  static constexpr HOST_INLINE uint64_t add_sub_limbs_64(const uint64_t* x, const uint64_t* y, uint64_t* r)
  {
    uint64_t carry = 0;
    carry_chain<NLIMBS, false, CARRY_OUT> chain;
    for (unsigned i = 0; i < NLIMBS / 2; i++)
      r[i] = SUBTRACT ? chain.sub(x[i], y[i], carry) : chain.add(x[i], y[i], carry);
    return CARRY_OUT ? carry : 0;
  }

  template <
    unsigned NLIMBS,
    bool SUBTRACT,
    bool CARRY_OUT,
    bool USE_32 = true> // for now we use only the 32 add/sub because the casting of the carry causes problems when
                        // compiling in release. to solve this we need to entirely split the field functions between a
                        // host version and a device version.
                        static constexpr HOST_INLINE uint32_t // 32 is enough for the carry
                        add_sub_limbs(const storage<NLIMBS>& xs, const storage<NLIMBS>& ys, storage<NLIMBS>& rs)
  {
    if constexpr (USE_32 || NLIMBS < 2) {
      const uint32_t* x = xs.limbs;
      const uint32_t* y = ys.limbs;
      uint32_t* r = rs.limbs;
      return add_sub_limbs_32<NLIMBS, SUBTRACT, CARRY_OUT>(x, y, r);
    } else {
      const uint64_t* x = xs.limbs64;
      const uint64_t* y = ys.limbs64;
      uint64_t* r = rs.limbs64;
      // Note: returns uint64 but uint 32 is enough.
      uint64_t result = add_sub_limbs_64<NLIMBS, SUBTRACT, CARRY_OUT>(x, y, r);
      uint32_t carry = result == 1;
      return carry;
    }
  }

  template <unsigned NLIMBS_A, unsigned NLIMBS_B = NLIMBS_A>
  static constexpr HOST_INLINE void
  multiply_raw_32(const storage<NLIMBS_A>& as, const storage<NLIMBS_B>& bs, storage<NLIMBS_A + NLIMBS_B>& rs)
  {
    const uint32_t* a = as.limbs;
    const uint32_t* b = bs.limbs;
    uint32_t* r = rs.limbs;
    for (unsigned i = 0; i < NLIMBS_B; i++) {
      uint32_t carry = 0;
      for (unsigned j = 0; j < NLIMBS_A; j++)
        r[j + i] = host_math::madc_cc(a[j], b[i], r[j + i], carry);
      r[NLIMBS_A + i] = carry;
    }
  }

  template <unsigned NLIMBS_A, unsigned NLIMBS_B = NLIMBS_A>
  static HOST_INLINE void multiply_raw_64(const uint64_t* a, const uint64_t* b, uint64_t* r)
  {
    for (unsigned i = 0; i < NLIMBS_B / 2; i++) {
      uint64_t carry = 0;
      for (unsigned j = 0; j < NLIMBS_A / 2; j++)
        r[j + i] = host_math::madc_cc_64(a[j], b[i], r[j + i], carry);
      r[NLIMBS_A / 2 + i] = carry;
    }
  }

  template <unsigned NLIMBS_A, unsigned NLIMBS_B = NLIMBS_A>
  static HOST_INLINE void
  multiply_raw_64(const storage<NLIMBS_A>& as, const storage<NLIMBS_B>& bs, storage<NLIMBS_A + NLIMBS_B>& rs)
  {
    const uint64_t* a = as.limbs64;
    const uint64_t* b = bs.limbs64;
    uint64_t* r = rs.limbs64;
    multiply_raw_64<NLIMBS_A, NLIMBS_B>(a, b, r);
  }

  template <unsigned NLIMBS_A, unsigned NLIMBS_B = NLIMBS_A, bool USE_32 = false>
  static constexpr HOST_INLINE void
  multiply_raw(const storage<NLIMBS_A>& as, const storage<NLIMBS_B>& bs, storage<NLIMBS_A + NLIMBS_B>& rs)
  {
    static_assert(
      (NLIMBS_A % 2 == 0 || NLIMBS_A == 1) && (NLIMBS_B % 2 == 0 || NLIMBS_B == 1),
      "odd number of limbs is not supported\n");
    if constexpr (USE_32) {
      multiply_raw_32<NLIMBS_A, NLIMBS_B>(as, bs, rs);
      return;
    } else if constexpr ((NLIMBS_A == 1 && NLIMBS_B == 2) || (NLIMBS_A == 2 && NLIMBS_B == 1)) {
      multiply_raw_32<NLIMBS_A, NLIMBS_B>(as, bs, rs);
      return;
    } else if constexpr (NLIMBS_A == 1 && NLIMBS_B == 1) {
      rs.limbs[1] = 0;
      rs.limbs[0] = host_math::madc_cc(as.limbs[0], bs.limbs[0], 0, rs.limbs[1]);
      return;
    } else if constexpr (NLIMBS_A == 2 && NLIMBS_B == 2) {
      const uint64_t* a = as.limbs64; // nof limbs should be even
      const uint64_t* b = bs.limbs64;
      uint64_t* r = rs.limbs64;
      r[1] = 0;
      r[0] = host_math::madc_cc_64(a[0], b[0], 0, r[1]);
      return;
    } else {
      multiply_raw_64<NLIMBS_A, NLIMBS_B>(as, bs, rs);
    }
  }

  template <unsigned NLIMBS, unsigned BITS>
  static constexpr HOST_INLINE storage<NLIMBS> left_shift(const storage<NLIMBS>& xs)
  {
    if constexpr (BITS == 0)
      return xs;
    else {
      constexpr unsigned BITS32 = BITS % 32;
      constexpr unsigned LIMBS_GAP = BITS / 32;
      storage<NLIMBS> out{};
      if constexpr (LIMBS_GAP < NLIMBS) {
        out.limbs[LIMBS_GAP] = xs.limbs[0] << BITS32;
        for (unsigned i = 1; i < NLIMBS - LIMBS_GAP; i++)
          out.limbs[i + LIMBS_GAP] = (xs.limbs[i] << BITS32) + (xs.limbs[i - 1] >> (32 - BITS32));
      }
      return out;
    }
  }

  template <unsigned NLIMBS, unsigned BITS>
  static constexpr HOST_INLINE storage<NLIMBS> right_shift(const storage<NLIMBS>& xs)
  {
    if constexpr (BITS == 0)
      return xs;
    else {
      constexpr unsigned BITS32 = BITS % 32;
      constexpr unsigned LIMBS_GAP = BITS / 32;
      storage<NLIMBS> out{};
      if constexpr (LIMBS_GAP < NLIMBS - 1) {
        for (unsigned i = 0; i < NLIMBS - LIMBS_GAP - 1; i++)
          out.limbs[i] = (xs.limbs[i + LIMBS_GAP] >> BITS32) + (xs.limbs[i + LIMBS_GAP + 1] << (32 - BITS32));
      }
      if constexpr (LIMBS_GAP < NLIMBS) out.limbs[NLIMBS - LIMBS_GAP - 1] = (xs.limbs[NLIMBS - 1] >> BITS32);
      return out;
    }
  }

  template <
    unsigned NLIMBS_NUM,
    unsigned NLIMBS_DENOM,
    unsigned NLIMBS_Q = (NLIMBS_NUM - NLIMBS_DENOM),
    bool USE_32 = false>
  static constexpr HOST_INLINE void integer_division(
    const storage<NLIMBS_NUM>& num, const storage<NLIMBS_DENOM>& denom, storage<NLIMBS_Q>& q, storage<NLIMBS_DENOM>& r)
  {
    storage<NLIMBS_DENOM> temp = {};
    for (int limb_idx = NLIMBS_NUM - 1; limb_idx >= 0; limb_idx--) {
      for (int bit_idx = 31; bit_idx >= 0; bit_idx--) {
        r = left_shift<NLIMBS_DENOM, 1>(r);
        r.limbs[0] |= ((num.limbs[limb_idx] >> bit_idx) & 1);
        uint32_t c = add_sub_limbs<NLIMBS_DENOM, true, true, USE_32>(r, denom, temp);
        if (limb_idx < NLIMBS_Q & !c) {
          r = temp;
          q.limbs[limb_idx] |= 1 << bit_idx;
        }
      }
    }
  }
} // namespace host_math

#pragma GCC reset_options