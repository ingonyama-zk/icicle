#pragma once

// Note: this optimization generates invalid code (using gcc) when storage class has a union for both u32 and u64 so
// disabling it.
#if defined(__GNUC__) && !defined(__NVCC__) && !defined(__clang__)
  #pragma GCC optimize("no-strict-aliasing")
#endif

#include <cstdint>
#include <cstring>
#include "icicle/utils/modifiers.h"
#include "icicle/math/storage.h"
#include "icicle/errors.h"
namespace host_math {

  // return x + y with T operands
  template <typename T>
  static constexpr HOST_INLINE T add(const T x, const T y)
  {
    return x + y;
  }

  // return x + y + carry with T operands
  template <typename T>
  static constexpr HOST_INLINE T addc(const T x, const T y, const T carry)
  {
    return x + y + carry;
  }

  // return x + y and carry out with T operands
  template <typename T>
  static constexpr HOST_INLINE T add_cc(const T x, const T y, T& carry)
  {
    T result = x + y;
    carry = x > result;
    return result;
  }

  template <>
  HOST_INLINE uint64_t add_cc(const uint64_t x, const uint64_t y, uint64_t& carry)
  {
    __uint128_t res_128 = static_cast<__uint128_t>(x) + y;
    carry = (uint64_t)(res_128 >> 64);
    uint64_t result = static_cast<uint64_t>(res_128);
    return result;
  }

  // return x + y + carry and carry out  with T operands
  template <typename T>
  static constexpr HOST_INLINE T addc_cc(const T x, const T y, T& carry)
  {
    const T result = x + y + carry;
    carry = carry && x >= result || !carry && x > result;
    return result;
  }

  template <>
  HOST_INLINE uint64_t addc_cc(const uint64_t x, const uint64_t y, uint64_t& carry)
  {
    __uint128_t res_128 = static_cast<__uint128_t>(x) + y + carry;
    carry = (uint64_t)(res_128 >> 64);
    uint64_t result = static_cast<uint64_t>(res_128);
    return result;
  }

  // return x - y with T operands
  template <typename T>
  static constexpr HOST_INLINE T sub(const T x, const T y)
  {
    return x - y;
  }

  //    return x - y - borrow with T operands
  template <typename T>
  static constexpr HOST_INLINE T subc(const T x, const T y, const T borrow)
  {
    return x - y - borrow;
  }

  //    return x - y and borrow out with T operands
  template <typename T>
  static constexpr HOST_INLINE T sub_cc(const T x, const T y, T& borrow)
  {
    T result = x - y;
    borrow = x < result;
    return result;
  }

  //    return x - y - borrow and borrow out with T operands
  template <typename T>
  static constexpr HOST_INLINE T subc_cc(const T x, const T y, T& borrow)
  {
    const T result = x - y - borrow;
    borrow = borrow && x <= result || !borrow && x < result;
    return result;
  }

  // return x * y + z + carry and carry out with uint32_t operands
  static constexpr HOST_INLINE uint32_t madc_cc(const uint32_t x, const uint32_t y, const uint32_t z, uint32_t& carry)
  {
    uint64_t r = static_cast<uint64_t>(x) * y + z + carry;
    carry = (uint32_t)(r >> 32);
    uint32_t result = r & 0xffffffff;
    return result;
  }

  static constexpr HOST_INLINE uint64_t
  madc_cc_64(const uint64_t x, const uint64_t y, const uint64_t z, uint64_t& carry)
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
#pragma unroll
    for (unsigned i = 0; i < NLIMBS; i++)
      r[i] = SUBTRACT ? chain.sub(x[i], y[i], carry) : chain.add(x[i], y[i], carry);
    return CARRY_OUT ? carry : 0;
  }

  template <unsigned NLIMBS, bool SUBTRACT, bool CARRY_OUT>
  static constexpr HOST_INLINE uint64_t add_sub_limbs_64(const uint64_t* x, const uint64_t* y, uint64_t* r)
  {
    uint64_t carry = 0;
    carry_chain<NLIMBS, false, CARRY_OUT> chain;
#pragma unroll
    for (unsigned i = 0; i < NLIMBS / 2; i++)
      r[i] = SUBTRACT ? chain.sub(x[i], y[i], carry) : chain.add(x[i], y[i], carry);
    return CARRY_OUT ? carry : 0;
  }

  // WARNING: taking views is zero copy but unsafe
  template <unsigned NLIMBS>
  constexpr const storage<NLIMBS>& get_lower_view(const storage<2 * NLIMBS>& xs)
  {
    return *reinterpret_cast<const storage<NLIMBS>*>(xs.limbs);
  }
  template <unsigned NLIMBS>
  constexpr const storage<NLIMBS>& get_higher_view(const storage<2 * NLIMBS>& xs)
  {
    return *reinterpret_cast<const storage<NLIMBS>*>(xs.limbs + NLIMBS);
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
#pragma unroll
    for (unsigned i = 0; i < NLIMBS_B; i++) {
      uint32_t carry = 0;
#pragma unroll
      for (unsigned j = 0; j < NLIMBS_A; j++)
        r[j + i] = host_math::madc_cc(a[j], b[i], r[j + i], carry);
      r[NLIMBS_A + i] = carry;
    }
  }

  template <unsigned NLIMBS_A, unsigned NLIMBS_B = NLIMBS_A>
  static HOST_INLINE void multiply_raw_64(const uint64_t* a, const uint64_t* b, uint64_t* r)
  {
#pragma unroll
    for (unsigned i = 0; i < NLIMBS_B / 2; i++) {
      uint64_t carry = 0;
#pragma unroll
      for (unsigned j = 0; j < NLIMBS_A / 2; j++) {
        r[j + i] = host_math::madc_cc_64(a[j], b[i], r[j + i], carry);
      }
      r[NLIMBS_A / 2 + i] = carry;
    }
  }

  // This multiplies only the LSB limbs and ignores the MSB ones so we output NLIMBS rather than 2*NLIMBS
  template <unsigned NLIMBS /*32b limbs*/>
  static HOST_INLINE void lsb_multiply_raw_64(const uint64_t* a, const uint64_t* b, uint64_t* r)
  {
#pragma unroll
    for (unsigned j = 0; j < NLIMBS / 2; j++) {
      uint64_t carry = 0;
#pragma unroll
      for (unsigned i = 0; i < NLIMBS / 2 - j; i++) {
        r[j + i] = host_math::madc_cc_64(a[j], b[i], r[j + i], carry);
      }
    }
  }

  template <unsigned NLIMBS_A, unsigned NLIMBS_B = NLIMBS_A, bool USE_32 = false>
  static constexpr HOST_INLINE void
  multiply_raw(const storage<NLIMBS_A>& as, const storage<NLIMBS_B>& bs, storage<NLIMBS_A + NLIMBS_B>& rs)
  {
    static_assert(
      ((NLIMBS_A % 2 == 0 || NLIMBS_A == 1) && (NLIMBS_B % 2 == 0 || NLIMBS_B == 1)) || USE_32,
      "odd number of limbs is not supported for 64 bit multiplication\n");
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
      multiply_raw_64<NLIMBS_A, NLIMBS_B>(as.limbs64, bs.limbs64, rs.limbs64);
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
#pragma unroll
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
#pragma unroll
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
#pragma unroll
    for (int limb_idx = NLIMBS_NUM - 1; limb_idx >= 0; limb_idx--) {
#pragma unroll
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
  template <unsigned NLIMBS, unsigned SLACK_BITS>
  static constexpr void get_higher_with_slack(const storage<2 * NLIMBS>& xs, storage<NLIMBS>& out)
  {
    // CPU: for even number of limbs, read and shift 64b limbs, otherwise 32b
    if constexpr (NLIMBS % 2 == 0) {
#pragma unroll
      for (unsigned i = 0; i < NLIMBS / 2; i++) { // Ensure valid indexing
        out.limbs64[i] =
          (xs.limbs64[i + NLIMBS / 2] << 2 * SLACK_BITS) | (xs.limbs64[i + NLIMBS / 2 - 1] >> (64 - 2 * SLACK_BITS));
      }
    } else {
#pragma unroll
      for (unsigned i = 0; i < NLIMBS; i++) { // Ensure valid indexing
        out.limbs[i] = (xs.limbs[i + NLIMBS] << 2 * SLACK_BITS) + (xs.limbs[i + NLIMBS - 1] >> (32 - 2 * SLACK_BITS));
      }
    }
  }
  template <unsigned NLIMBS>
  static constexpr bool is_equal(const storage<NLIMBS>& xs, const storage<NLIMBS>& ys)
  {
    return std::memcmp(xs.limbs, ys.limbs, NLIMBS * sizeof(xs.limbs[0])) == 0;
  }
  // this function checks if the given index is within the array range
  static constexpr void index_err(uint32_t index, uint32_t max_index)
  {
    if (index > max_index)
      THROW_ICICLE_ERR(
        icicle::eIcicleError::INVALID_ARGUMENT, "Field: index out of range: given index -" + std::to_string(index) +
                                                  "> max index - " + std::to_string(max_index));
  }

  template <unsigned NLIMBS>
  static constexpr void multiply_and_add_lsb_neg_modulus_raw(
    const storage<NLIMBS>& as, const storage<NLIMBS>& neg_mod, const storage<NLIMBS>& cs, storage<NLIMBS>& rs)
  {
    // NOTE: we need an LSB-multiplier here so it's inefficient to do a full multiplier. Having said that it
    // seems that after optimization (inlining probably), the compiler eliminates the msb limbs since they are unused.
    // The following code is not assuming so and uses an LSB-multiplier explicitly (although they perform the same for
    // optimized code, but not for debug).
    if constexpr (NLIMBS > 1) {
      // LSB multiplier, computed only NLIMBS output limbs
      storage<NLIMBS> r_low = {};
      lsb_multiply_raw_64<NLIMBS>(as.limbs64, neg_mod.limbs64, r_low.limbs64);
      add_sub_limbs<NLIMBS, false, false>(cs, r_low, rs);
    } else {
      // case of one limb is using a single 32b multiplier anyway
      storage<2 * NLIMBS> r_wide = {};
      multiply_raw<NLIMBS>(as, neg_mod, r_wide);
      const storage<NLIMBS>& r_low_view = get_lower_view<NLIMBS>(r_wide);
      add_sub_limbs<NLIMBS, false, false>(cs, r_low_view, rs);
    }
  }

  /**
   * This method reduces a Wide number `xs` modulo `p` and returns the result as a Field element.
   *
   * It is assumed that the high `2 * slack_bits` bits of `xs` are unset which is always the case for the product of 2
   * numbers with their high `slack_bits` unset. Larger Wide numbers should be reduced by subtracting an appropriate
   * factor of `modulus_squared` first.
   *
   * This function implements ["multi-precision Barrett"](https://github.com/ingonyama-zk/modular_multiplication). As
   * opposed to Montgomery reduction, it doesn't require numbers to have a special representation but lets us work with
   * them as-is. The general idea of Barrett reduction is to estimate the quotient \f$ l \approx \floor{\frac{xs}{p}}
   * \f$ and return \f$ xs - l \cdot p \f$. But since \f$ l \f$ is inevitably computed with an error (it's always less
   * or equal than the real quotient). So the modulus `p` might need to be subtracted several times before the result is
   * in the desired range \f$ [0;p-1] \f$. The estimate of the error is as follows: \f[ \frac{xs}{p} - l = \frac{xs}{p}
   * - \frac{xs \cdot m}{2^{2n}} + \frac{xs \cdot m}{2^{2n}} - \floor{\frac{xs}{2^k}}\frac{m}{2^{2n-k}}
   *  + \floor{\frac{xs}{2^k}}\frac{m}{2^{2n-k}} - l \leq p^2(\frac{1}{p}-\frac{m}{2^{2n}}) + \frac{m}{2^{2n-k}} + 2(TLC
   * - 1) \cdot 2^{-32} \f] Here \f$ l \f$ is the result of [multiply_msb_raw](@ref multiply_msb_raw) function and the
   * last term in the error is due to its approximation. \f$ n \f$ is the number of bits in \f$ p \f$ and \f$ k = 2n -
   * 32\cdot TLC \f$. Overall, the error is always less than 2 so at most 2 reductions are needed. However, in most
   * cases it's less than 1, so setting the [num_of_reductions](@ref num_of_reductions) variable for a field equal to 1
   * will cause only 1 reduction to be performed.
   */
  template <unsigned NLIMBS, unsigned SLACK_BITS, unsigned NOF_REDUCTIONS>
  static constexpr HOST_DEVICE_INLINE storage<NLIMBS> barrett_reduce(
    const storage<2 * NLIMBS>& xs,
    const storage<NLIMBS>& ms,
    const storage<NLIMBS>& mod1,
    const storage<NLIMBS>& mod2,
    const storage<NLIMBS>& neg_mod)
  {
    storage<2 * NLIMBS> l = {}; // the approximation of l for a*b = l*p + r mod p
    storage<NLIMBS> r = {};

    // `xs` is left-shifted by `2 * slack_bits` and higher half is written to `xs_hi`
    storage<NLIMBS> xs_hi = {};
    get_higher_with_slack<NLIMBS, SLACK_BITS>(xs, xs_hi);
    multiply_raw<NLIMBS>(xs_hi, ms, l); // MSB mult by `m`. TODO - msb optimization
    // Note: taking views is zero copy but unsafe
    const storage<NLIMBS>& l_hi = get_higher_view<NLIMBS>(l);
    const storage<NLIMBS>& xs_lo = get_lower_view<NLIMBS>(xs);
    // Here we need to compute the lsb of `xs - l \cdot p` and to make use of fused multiply-and-add, we rewrite it as
    // `xs + l \cdot (2^{32 \cdot TLC}-p)` which is the same as original (up to higher limbs which we don't care about).
    multiply_and_add_lsb_neg_modulus_raw(l_hi, neg_mod, xs_lo, r);
    // As mentioned, either 2 or 1 reduction can be performed depending on the field in question.
    if constexpr (NOF_REDUCTIONS == 2) {
      storage<NLIMBS> r_reduced = {};
      const auto borrow = add_sub_limbs<NLIMBS, true, true>(r, mod2, r_reduced);
      // If r-2p has no borrow then we are done
      if (!borrow) return r_reduced;
    }
    // if r-2p has borrow then we need to either subtract p or we are already in [0,p).
    // so we subtract p and based on the borrow bit we know which case it is
    storage<NLIMBS> r_reduced = {};
    const auto borrow = add_sub_limbs<NLIMBS, true, true>(r, mod1, r_reduced);
    return borrow ? r : r_reduced;
  }

  // Assumes the number is even!
  template <unsigned NLIMBS>
  static constexpr void div2(const storage<NLIMBS>& xs, storage<NLIMBS>& rs)
  {
    // Note: volatile is used to prevent compiler optimizations that assume strict aliasing rules.
    volatile const uint32_t* x = xs.limbs;
    volatile uint32_t* r = rs.limbs;
    if constexpr (NLIMBS > 1) {
      for (unsigned i = 0; i < NLIMBS - 1; i++) {
        r[i] = (x[i] >> 1) | (x[i + 1] << 31);
      }
    }
    r[NLIMBS - 1] = x[NLIMBS - 1] >> 1;
  }
} // namespace host_math

#if defined(__GNUC__) && !defined(__NVCC__) && !defined(__clang__)
  #pragma GCC reset_options
#endif
