#pragma once

// Note: this optimization generates invalid code (using gcc) when storage class has a union for both u32 and u64 so
// disabling it.
#if defined(__GNUC__) && !defined(__NVCC__) && !defined(__clang__)
  #pragma GCC optimize("no-strict-aliasing")
#endif

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

  static __host__ uint64_t madc_cc_64(const uint64_t x, const uint64_t y, const uint64_t z, uint64_t& carry)
  {
    __uint128_t r = static_cast<__uint128_t>(x) * y + z + carry;

    carry = (uint64_t)(r >> 64);
    uint64_t result = r & 0xffffffffffffffff;
    return result;
  }

#include <cstdint>

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
    bool USE_32 = false>
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
  static constexpr HOST_INLINE void
  multiply_mont_32(const uint32_t* a, const uint32_t* b, const uint32_t* q, const uint32_t* p, uint32_t* r)
  {
    for (unsigned i = 0; i < NLIMBS_B; i++) {
      uint32_t A = 0, C = 0;
      r[0] = host_math::madc_cc(a[0], b[i], r[0], A);
      uint32_t m = host_math::madc_cc(r[0], q[0], 0, C); // TODO - multiply inst
      C = 0;
      host_math::madc_cc(m, p[0], r[0], C);
      for (unsigned j = 1; j < NLIMBS_A; j++) {
        r[j] = host_math::madc_cc(a[j], b[i], r[j], A);
        r[j - 1] = host_math::madc_cc(m, p[j], r[j], C);
      }
      r[NLIMBS_A - 1] = C + A;
    }
  }

  template <unsigned NLIMBS_A, unsigned NLIMBS_B = NLIMBS_A>
  static HOST_INLINE void
  multiply_mont_64(const uint64_t* a, const uint64_t* b, const uint64_t* q, const uint64_t* p, uint64_t* r)
  {
    for (unsigned i = 0; i < NLIMBS_B / 2; i++) {
      uint64_t A = 0, C = 0;
      r[0] = host_math::madc_cc_64(a[0], b[i], r[0], A);
      uint64_t m = host_math::madc_cc_64(r[0], q[0], 0, C); // TODO - multiply inst
      C = 0;
      host_math::madc_cc_64(m, p[0], r[0], C);
      for (unsigned j = 1; j < NLIMBS_A / 2; j++) {
        r[j] = host_math::madc_cc_64(a[j], b[i], r[j], A);
        r[j - 1] = host_math::madc_cc_64(m, p[j], r[j], C);
      }
      r[NLIMBS_A / 2 - 1] = C + A;
    }
  }

  /**
   * @brief Perform  SOS reduction on a number in montgomery representation \p t in range [0, \p n ^2-1] limiting it to
   * the range [0,2 \p n -1].
   * @param t Number to be reduced. Must be in montgomery rep, and in range [0, \p n ^2-1].
   * @param n Field modulus.
   * @param n_tag Number such that \p n * \p n_tag modR = -1
   * @param r Array in which to store the result in its upper half (Lower half is data that would be removed by
   * dividing by R = shifting NLIMBS down).
   * @tparam NLIMBS Number of 32bit limbs required to represent a number in the field defined by n. R is 2^(NLIMBS*32).
   */
  template <unsigned NLIMBS>
  static HOST_INLINE void
  sos_mont_reduction_32(const uint32_t* t, const uint32_t* n, const uint32_t* n_tag, uint32_t* r)
  {
    const unsigned s = NLIMBS; // For similarity to the original algorithm

    // Copy t to r as t is read-only
    for (int i = 0; i < 2 * s; i++) {
      r[i] = t[i];
    }

    for (int i = 0; i < s; i++) {
      uint32_t c = 0;
      uint32_t m = r[i] * n_tag[0];

      for (int j = 0; j < s; j++) {
        // r[i+j] = addc_cc(r[i+j], m * n[j], c);
        r[i + j] = madc_cc(m, n[j], r[i + j], c);
      }
      // Propagate the carry to the remaining sublimbs
      for (int carry_idx = s + i; carry_idx < 2 * s; carry_idx++) {
        if (c == 0) { break; }
        r[carry_idx] = add_cc(r[carry_idx], c, c);
      }
    }
  }

  /**
   * @brief Perform  SOS reduction on a number in montgomery representation \p t in range [0, \p n ^2-1] limiting it to
   * the range [0,2 \p n -1].
   * @param t Number to be reduced. Must be in montgomery rep, and in range [0, \p n ^2-1].
   * @param n Field modulus.
   * @param n_tag Number such that \p n * \p n_tag modR = -1
   * @param r Array in which to store the result in its upper half (Lower half is data that would be removed by
   * dividing by R = shifting NLIMBS down).
   * @tparam NLIMBS Number of 32bit limbs required to represent a number in the field defined by n. R is 2^(NLIMBS*32).
   */
  template <unsigned NLIMBS>
  static HOST_INLINE void
  sos_mont_reduction_64(const uint64_t* t, const uint64_t* n, const uint64_t* n_tag, uint64_t* r)
  {
    const unsigned s = NLIMBS / 2; // Divide by 2 because NLIMBS is 32bit and this function is 64bit

    // Copy t to r as t is read-only
    for (int i = 0; i < 2 * s; i++) {
      r[i] = t[i];
    }

    for (int i = 0; i < s; i++) {
      uint64_t c = 0;
      uint64_t m = r[i] * n_tag[0];

      for (int j = 0; j < s; j++) {
        // r[i+j] = addc_cc(r[i+j], m * n[j], c);
        r[i + j] = madc_cc_64(m, n[j], r[i + j], c);
      }
      // Propagate the carry to the remaining sublimbs
      for (int carry_idx = s + i; carry_idx < 2 * s; carry_idx++) {
        if (c == 0) { break; }
        r[carry_idx] = add_cc(r[carry_idx], c, c);
      }
    }
  }

  /**
   * @brief Perform  SOS reduction on a number in montgomery representation \p t in range [0, \p n ^2-1] limiting it to
   * the range [0,2 \p n -1].
   * @param t Number to be reduced. Must be in montgomery rep, and in range [0, \p n ^2-1].
   * @param n Field modulus.
   * @param n_tag Number such that \p n * \p n_tag modR = -1
   * @param r Array in which to store the result in its upper half (Lower half is data that would be removed by
   * dividing by R = shifting NLIMBS down).
   * @tparam NLIMBS Number of 32bit limbs required to represent a number in the field defined by n. R is 2^(NLIMBS*32).
   */
  template <unsigned NLIMBS, bool USE_32 = false>
  static HOST_INLINE void sos_mont_reduction(
    const storage<2 * NLIMBS>& t, const storage<NLIMBS>& n, const storage<NLIMBS>& n_tag, storage<2 * NLIMBS>& r)
  {
    static_assert(NLIMBS % 2 == 0 || NLIMBS == 1, "Odd number of limbs (That is not 1) is not supported\n");
    if constexpr (USE_32) {
      sos_mont_reduction_32<NLIMBS>(t.limbs, n.limbs, n_tag.limbs, r.limbs);
      return;
    } else if constexpr (NLIMBS == 1) {
      sos_mont_reduction_32<NLIMBS>(t.limbs, n.limbs, n_tag.limbs, r.limbs);
      return;
    } else {
      sos_mont_reduction_64<NLIMBS>(t.limbs64, n.limbs64, n_tag.limbs64, r.limbs64);
    }
  }

  template <unsigned NLIMBS_A, unsigned NLIMBS_B = NLIMBS_A, bool USE_32 = false>
  static constexpr HOST_INLINE void multiply_mont(
    const storage<NLIMBS_A>& as,
    const storage<NLIMBS_B>& bs,
    const storage<NLIMBS_A>& qs,
    const storage<NLIMBS_A>& ps,
    storage<NLIMBS_A>& rs)
  {
    static_assert(
      (NLIMBS_A % 2 == 0 || NLIMBS_A == 1) && (NLIMBS_B % 2 == 0 || NLIMBS_B == 1),
      "odd number of limbs is not supported\n");
    if constexpr (USE_32) {
      multiply_mont_32<NLIMBS_A, NLIMBS_B>(as.limbs, bs.limbs, qs.limbs, ps.limbs, rs.limbs);
      return;
    } else if constexpr (NLIMBS_A == 1 || NLIMBS_B == 1) {
      multiply_mont_32<NLIMBS_A, NLIMBS_B>(as.limbs, bs.limbs, qs.limbs, ps.limbs, rs.limbs);
      return;
    } else {
      multiply_mont_64<NLIMBS_A, NLIMBS_B>(as.limbs64, bs.limbs64, qs.limbs64, ps.limbs64, rs.limbs64);
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
        if ((limb_idx < NLIMBS_Q) & !c) {
          r = temp;
          q.limbs[limb_idx] |= 1 << bit_idx;
        }
      }
    }
  }
  template <unsigned NLIMBS>
  static constexpr void get_higher_with_slack(const storage<2 * NLIMBS>& xs, storage<NLIMBS>& out, unsigned slack_bits)
  {
    for (unsigned i = 0; i < NLIMBS; i++) {
      out.limbs[i] = (xs.limbs[i + NLIMBS] << 2 * slack_bits) + (xs.limbs[i + NLIMBS - 1] >> (32 - 2 * slack_bits));
    }
  }

  template <unsigned NLIMBS>
  static constexpr bool is_equal(const storage<NLIMBS>& xs, const storage<NLIMBS>& ys)
  {
    for (unsigned i = 0; i < NLIMBS; i++)
      if (xs.limbs[i] != ys.limbs[i]) return false;
    return true;
  }
} // namespace host_math

#if defined(__GNUC__) && !defined(__NVCC__) && !defined(__clang__)
  #pragma GCC reset_options
#endif
