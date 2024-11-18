#ifdef __CUDACC__

#pragma once

#include <cstdint>
#include "icicle/utils/modifiers.h"
#include "icicle/fields/storage.h"
#include "ptx.h"

namespace device_math {

template <unsigned OPS_COUNT = UINT32_MAX, bool CARRY_IN = false, bool CARRY_OUT = false> 
struct carry_chain {
  unsigned index;

  constexpr __device__ __forceinline__ carry_chain() : index(0) {}

  __device__ __forceinline__ uint32_t add(const uint32_t x, const uint32_t y) {
    index++;
    if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
      return ptx::add(x, y);
    else if (index == 1 && !CARRY_IN)
      return ptx::add_cc(x, y);
    else if (index < OPS_COUNT || CARRY_OUT)
      return ptx::addc_cc(x, y);
    else
      return ptx::addc(x, y);
  }

  __device__ __forceinline__ uint32_t sub(const uint32_t x, const uint32_t y) {
    index++;
    if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
      return ptx::sub(x, y);
    else if (index == 1 && !CARRY_IN)
      return ptx::sub_cc(x, y);
    else if (index < OPS_COUNT || CARRY_OUT)
      return ptx::subc_cc(x, y);
    else
      return ptx::subc(x, y);
  }

  __device__ __forceinline__ uint32_t mad_lo(const uint32_t x, const uint32_t y, const uint32_t z) {
    index++;
    if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
      return ptx::mad_lo(x, y, z);
    else if (index == 1 && !CARRY_IN)
      return ptx::mad_lo_cc(x, y, z);
    else if (index < OPS_COUNT || CARRY_OUT)
      return ptx::madc_lo_cc(x, y, z);
    else
      return ptx::madc_lo(x, y, z);
  }

  __device__ __forceinline__ uint32_t mad_hi(const uint32_t x, const uint32_t y, const uint32_t z) {
    index++;
    if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
      return ptx::mad_hi(x, y, z);
    else if (index == 1 && !CARRY_IN)
      return ptx::mad_hi_cc(x, y, z);
    else if (index < OPS_COUNT || CARRY_OUT)
      return ptx::madc_hi_cc(x, y, z);
    else
      return ptx::madc_hi(x, y, z);
  }
};

template <unsigned NLIMBS, bool SUBTRACT, bool CARRY_OUT>
  static constexpr DEVICE_INLINE uint32_t add_sub_u32_device(const uint32_t* x, const uint32_t* y, uint32_t* r)
  {
    r[0] = SUBTRACT ? ptx::sub_cc(x[0], y[0]) : ptx::add_cc(x[0], y[0]);
    for (unsigned i = 1; i < NLIMBS; i++)
      r[i] = SUBTRACT ? ptx::subc_cc(x[i], y[i]) : ptx::addc_cc(x[i], y[i]);
    if (!CARRY_OUT) {
      ptx::addc(0, 0);
      return 0;
    }
    return SUBTRACT ? ptx::subc(0, 0) : ptx::addc(0, 0);
  }

  template <unsigned NLIMBS, bool SUBTRACT, bool CARRY_OUT>
  static constexpr DEVICE_INLINE uint32_t
  add_sub_limbs_device(const storage<NLIMBS>& xs, const storage<NLIMBS>& ys, storage<NLIMBS>& rs)
  {
    const uint32_t* x = xs.limbs;
    const uint32_t* y = ys.limbs;
    uint32_t* r = rs.limbs;
    return add_sub_u32_device<NLIMBS, SUBTRACT, CARRY_OUT>(x, y, r);
  }

  template <unsigned NLIMBS>
  static DEVICE_INLINE void mul_n(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = NLIMBS)
  {
    UNROLL
    for (size_t i = 0; i < n; i += 2) {
      acc[i] = ptx::mul_lo(a[i], bi);
      acc[i + 1] = ptx::mul_hi(a[i], bi);
    }
  }

  template <unsigned NLIMBS>
  static DEVICE_INLINE void mul_n_msb(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = NLIMBS, size_t start_i = 0)
  {
    UNROLL
    for (size_t i = start_i; i < n; i += 2) {
      acc[i] = ptx::mul_lo(a[i], bi);
      acc[i + 1] = ptx::mul_hi(a[i], bi);
    }
  }

  template <unsigned NLIMBS, bool CARRY_IN = false>
  static DEVICE_INLINE void
  cmad_n(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = NLIMBS, uint32_t optional_carry = 0)
  {
    if (CARRY_IN) ptx::add_cc(UINT32_MAX, optional_carry);
    acc[0] = CARRY_IN ? ptx::madc_lo_cc(a[0], bi, acc[0]) : ptx::mad_lo_cc(a[0], bi, acc[0]);
    acc[1] = ptx::madc_hi_cc(a[0], bi, acc[1]);

    UNROLL
    for (size_t i = 2; i < n; i += 2) {
      acc[i] = ptx::madc_lo_cc(a[i], bi, acc[i]);
      acc[i + 1] = ptx::madc_hi_cc(a[i], bi, acc[i + 1]);
    }
  }

  template <unsigned NLIMBS, bool EVEN_PHASE>
  static DEVICE_INLINE void cmad_n_msb(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = NLIMBS)
  {
    if (EVEN_PHASE) {
      acc[0] = ptx::mad_lo_cc(a[0], bi, acc[0]);
      acc[1] = ptx::madc_hi_cc(a[0], bi, acc[1]);
    } else {
      acc[1] = ptx::mad_hi_cc(a[0], bi, acc[1]);
    }

    UNROLL
    for (size_t i = 2; i < n; i += 2) {
      acc[i] = ptx::madc_lo_cc(a[i], bi, acc[i]);
      acc[i + 1] = ptx::madc_hi_cc(a[i], bi, acc[i + 1]);
    }
  }

  template <unsigned NLIMBS>
  static DEVICE_INLINE void cmad_n_lsb(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = NLIMBS)
  {
    if (n > 1)
      acc[0] = ptx::mad_lo_cc(a[0], bi, acc[0]);
    else
      acc[0] = ptx::mad_lo(a[0], bi, acc[0]);

    size_t i;
    UNROLL
    for (i = 1; i < n - 1; i += 2) {
      acc[i] = ptx::madc_hi_cc(a[i - 1], bi, acc[i]);
      if (i == n - 2)
        acc[i + 1] = ptx::madc_lo(a[i + 1], bi, acc[i + 1]);
      else
        acc[i + 1] = ptx::madc_lo_cc(a[i + 1], bi, acc[i + 1]);
    }
    if (i == n - 1) acc[i] = ptx::madc_hi(a[i - 1], bi, acc[i]);
  }

  template <unsigned NLIMBS, bool CARRY_OUT = false, bool CARRY_IN = false>
  static DEVICE_INLINE uint32_t mad_row(
    uint32_t* odd,
    uint32_t* even,
    const uint32_t* a,
    uint32_t bi,
    size_t n = NLIMBS,
    uint32_t ci = 0,
    uint32_t di = 0,
    uint32_t carry_for_high = 0,
    uint32_t carry_for_low = 0)
  {
    cmad_n<NLIMBS, CARRY_IN>(odd, a + 1, bi, n - 2, carry_for_low);
    odd[n - 2] = ptx::madc_lo_cc(a[n - 1], bi, ci);
    odd[n - 1] = CARRY_OUT ? ptx::madc_hi_cc(a[n - 1], bi, di) : ptx::madc_hi(a[n - 1], bi, di);
    uint32_t cr = CARRY_OUT ? ptx::addc(0, 0) : 0;
    cmad_n<NLIMBS>(even, a, bi, n);
    if (CARRY_OUT) {
      odd[n - 1] = ptx::addc_cc(odd[n - 1], carry_for_high);
      cr = ptx::addc(cr, 0);
    } else
      odd[n - 1] = ptx::addc(odd[n - 1], carry_for_high);
    return cr;
  }

  template <unsigned NLIMBS, bool EVEN_PHASE>
  static DEVICE_INLINE void mad_row_msb(uint32_t* odd, uint32_t* even, const uint32_t* a, uint32_t bi, size_t n = NLIMBS)
  {
    cmad_n_msb<NLIMBS,!EVEN_PHASE>(odd, EVEN_PHASE ? a : (a + 1), bi, n - 2);
    odd[EVEN_PHASE ? (n - 1) : (n - 2)] = ptx::madc_lo_cc(a[n - 1], bi, 0);
    odd[EVEN_PHASE ? n : (n - 1)] = ptx::madc_hi(a[n - 1], bi, 0);
    cmad_n_msb<NLIMBS,EVEN_PHASE>(even, EVEN_PHASE ? (a + 1) : a, bi, n - 1);
    odd[EVEN_PHASE ? n : (n - 1)] = ptx::addc(odd[EVEN_PHASE ? n : (n - 1)], 0);
  }

  template <unsigned NLIMBS>
  static DEVICE_INLINE void mad_row_lsb(uint32_t* odd, uint32_t* even, const uint32_t* a, uint32_t bi, size_t n = NLIMBS)
  {
    // bi here is constant so we can do a compile-time check for zero (which does happen once for bls12-381 scalar field
    // modulus)
    if (bi != 0) {
      if (n > 1) cmad_n_lsb<NLIMBS>(odd, a + 1, bi, n - 1);
      cmad_n_lsb<NLIMBS>(even, a, bi, n);
    }
    return;
  }

  template <unsigned NLIMBS>
  static DEVICE_INLINE uint32_t
  mul_n_and_add(uint32_t* acc, const uint32_t* a, uint32_t bi, uint32_t* extra, size_t n = (NLIMBS >> 1))
  {
    acc[0] = ptx::mad_lo_cc(a[0], bi, extra[0]);

    UNROLL
    for (size_t i = 1; i < n - 1; i += 2) {
      acc[i] = ptx::madc_hi_cc(a[i - 1], bi, extra[i]);
      acc[i + 1] = ptx::madc_lo_cc(a[i + 1], bi, extra[i + 1]);
    }

    acc[n - 1] = ptx::madc_hi_cc(a[n - 2], bi, extra[n - 1]);
    return ptx::addc(0, 0);
  }

  /**
   * This method multiplies `a` and `b` (both assumed to have NLIMBS / 2 limbs) and adds `in1` and `in2` (NLIMBS limbs each)
   * to the result which is written to `even`.
   *
   * It is used to compute the "middle" part of Karatsuba: \f$ a_{lo} \cdot b_{hi} + b_{lo} \cdot a_{hi} =
   * (a_{hi} - a_{lo})(b_{lo} - b_{hi}) + a_{lo} \cdot b_{lo} + a_{hi} \cdot b_{hi} \f$. Currently this method assumes
   * that the top bit of \f$ a_{hi} \f$ and \f$ b_{hi} \f$ are unset. This ensures correctness by allowing to keep the
   * result inside NLIMBS limbs and ignore the carries from the highest limb.
   */
  template <unsigned NLIMBS>
  static DEVICE_INLINE void
  multiply_and_add_short_raw_device(const uint32_t* a, const uint32_t* b, uint32_t* even, uint32_t* in1, uint32_t* in2)
  {
    __align__(16) uint32_t odd[NLIMBS - 2];
    uint32_t first_row_carry = mul_n_and_add<NLIMBS>(even, a, b[0], in1);
    uint32_t carry = mul_n_and_add<NLIMBS>(odd, a + 1, b[0], &in2[1]);

    size_t i;
    UNROLL
    for (i = 2; i < ((NLIMBS >> 1) - 1); i += 2) {
      carry = mad_row<NLIMBS, true, false>(
        &even[i], &odd[i - 2], a, b[i - 1], NLIMBS >> 1, in1[(NLIMBS >> 1) + i - 2], in1[(NLIMBS >> 1) + i - 1], carry);
      carry =
        mad_row<NLIMBS, true, false>(&odd[i], &even[i], a, b[i], NLIMBS >> 1, in2[(NLIMBS >> 1) + i - 1], in2[(NLIMBS >> 1) + i], carry);
    }
    mad_row<NLIMBS, false, true>(
      &even[NLIMBS >> 1], &odd[(NLIMBS >> 1) - 2], a, b[(NLIMBS >> 1) - 1], NLIMBS >> 1, in1[NLIMBS - 2], in1[NLIMBS - 1], carry,
      first_row_carry);
    // merge |even| and |odd| plus the parts of `in2` we haven't added yet (first and last limbs)
    even[0] = ptx::add_cc(even[0], in2[0]);
    for (i = 0; i < (NLIMBS - 2); i++)
      even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
    even[i + 1] = ptx::addc(even[i + 1], in2[i + 1]);
  }



    /**
   * This method multiplies `a` and `b` and writes the result into `even`. It assumes that `a` and `b` are NLIMBS/2 limbs
   * long. The usual schoolbook algorithm is used.
   */
  template <unsigned NLIMBS>
  static DEVICE_INLINE void multiply_short_raw_device(const uint32_t* a, const uint32_t* b, uint32_t* even)
  {
    __align__(16) uint32_t odd[NLIMBS - 2];
    mul_n<NLIMBS>(even, a, b[0], NLIMBS >> 1);
    mul_n<NLIMBS>(odd, a + 1, b[0], NLIMBS >> 1);
    mad_row<NLIMBS>(&even[2], &odd[0], a, b[1], NLIMBS >> 1);

    size_t i;
    UNROLL
    for (i = 2; i < ((NLIMBS >> 1) - 1); i += 2) {
      mad_row<NLIMBS>(&odd[i], &even[i], a, b[i], NLIMBS >> 1);
      mad_row<NLIMBS>(&even[i + 2], &odd[i], a, b[i + 1], NLIMBS >> 1);
    }
    // merge |even| and |odd|
    even[1] = ptx::add_cc(even[1], odd[0]);
    for (i = 1; i < NLIMBS - 2; i++)
      even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
    even[i + 1] = ptx::addc(even[i + 1], 0);
  }

  /**
   * This method multiplies `as` and `bs` and writes the (wide) result into `rs`.
   *
   * It is assumed that the highest bits of `as` and `bs` are unset which is true for all the numbers icicle had to deal
   * with so far. This method implements [subtractive
   * Karatsuba](https://en.wikipedia.org/wiki/Karatsuba_algorithm#Implementation).
   */
  template <unsigned NLIMBS>
  static DEVICE_INLINE void multiply_raw_device(const storage<NLIMBS>& as, const storage<NLIMBS>& bs,  storage<2*NLIMBS>& rs)
  {
    const uint32_t* a = as.limbs;
    const uint32_t* b = bs.limbs;
    uint32_t* r = rs.limbs;
    if constexpr (NLIMBS > 2) {
      // Next two lines multiply high and low halves of operands (\f$ a_{lo} \cdot b_{lo}; a_{hi} \cdot b_{hi} \$f) and
      // write the results into `r`.
      multiply_short_raw_device<NLIMBS>(a, b, r);
      multiply_short_raw_device<NLIMBS>(&a[NLIMBS >> 1], &b[NLIMBS >> 1], &r[NLIMBS]);
      __align__(16) uint32_t middle_part[NLIMBS];
      __align__(16) uint32_t diffs[NLIMBS];
      // Differences of halves \f$ a_{hi} - a_{lo}; b_{lo} - b_{hi} \$f are written into `diffs`, signs written to
      // `carry1` and `carry2`.
      uint32_t carry1 = add_sub_u32_device<(NLIMBS >> 1), true, true>(&a[NLIMBS >> 1], a, diffs);
      uint32_t carry2 = add_sub_u32_device<(NLIMBS >> 1), true, true>(b, &b[NLIMBS >> 1], &diffs[NLIMBS >> 1]);
      // Compute the "middle part" of Karatsuba: \f$ a_{lo} \cdot b_{hi} + b_{lo} \cdot a_{hi} \f$.
      // This is where the assumption about unset high bit of `a` and `b` is relevant.
      multiply_and_add_short_raw_device<NLIMBS>(diffs, &diffs[NLIMBS >> 1], middle_part, r, &r[NLIMBS]);
      // Corrections that need to be performed when differences are negative.
      // Again, carry doesn't need to be propagated due to unset high bits of `a` and `b`.
      if (carry1)
        add_sub_u32_device<(NLIMBS >> 1), true, false>(&middle_part[NLIMBS >> 1], &diffs[NLIMBS >> 1], &middle_part[NLIMBS >> 1]);
      if (carry2) add_sub_u32_device<(NLIMBS >> 1), true, false>(&middle_part[NLIMBS >> 1], diffs, &middle_part[NLIMBS >> 1]);
      // Now that middle part is fully correct, it can be added to the result.
      add_sub_u32_device<NLIMBS, false, true>(&r[NLIMBS >> 1], middle_part, &r[NLIMBS >> 1]);

      // Carry from adding middle part has to be propagated to the highest limb.
      for (size_t i = NLIMBS + (NLIMBS >> 1); i < 2 * NLIMBS; i++)
        r[i] = ptx::addc_cc(r[i], 0);
    } else if (NLIMBS == 2) {
      __align__(8) uint32_t odd[2];
      r[0] = ptx::mul_lo(a[0], b[0]);
      r[1] = ptx::mul_hi(a[0], b[0]);
      r[2] = ptx::mul_lo(a[1], b[1]);
      r[3] = ptx::mul_hi(a[1], b[1]);
      odd[0] = ptx::mul_lo(a[0], b[1]);
      odd[1] = ptx::mul_hi(a[0], b[1]);
      odd[0] = ptx::mad_lo(a[1], b[0], odd[0]);
      odd[1] = ptx::mad_hi(a[1], b[0], odd[1]);
      r[1] = ptx::add_cc(r[1], odd[0]);
      r[2] = ptx::addc_cc(r[2], odd[1]);
      r[3] = ptx::addc(r[3], 0);
    } else if (NLIMBS == 1) {
      r[0] = ptx::mul_lo(a[0], b[0]);
      r[1] = ptx::mul_hi(a[0], b[0]);
    }
  }

  /**
   * A function that computes wide product \f$ rs = as \cdot bs \f$ that's correct for the higher NLIMBS + 1 limbs with a
   * small maximum error.
   *
   * The way this function saves computations (as compared to regular school-book multiplication) is by not including
   * terms that are too small. Namely, limb product \f$ a_i \cdot b_j \f$ is excluded if \f$ i + j < NLIMBS - 2 \f$ and
   * only the higher half is included if \f$ i + j = NLIMBS - 2 \f$. All other limb products are included. So, the error
   * i.e. difference between true product and the result of this function written to `rs` is exactly the sum of all
   * dropped limbs products, which we can bound: \f$ a_0 \cdot b_0 + 2^{32}(a_0 \cdot b_1 + a_1 \cdot b_0) + \dots +
   * 2^{32(NLIMBS - 3)}(a_{NLIMBS - 3} \cdot b_0 + \dots + a_0 \cdot b_{NLIMBS - 3}) + 2^{32(NLIMBS - 2)}(\floor{\frac{a_{NLIMBS - 2}
   * \cdot b_0}{2^{32}}} + \dots + \floor{\frac{a_0 \cdot b_{NLIMBS - 2}}{2^{32}}}) \leq 2^{64} + 2\cdot 2^{96} + \dots +
   * (NLIMBS - 2) \cdot 2^{32(NLIMBS - 1)} + (NLIMBS - 1) \cdot 2^{32(NLIMBS - 1)} \leq 2(NLIMBS - 1) \cdot 2^{32(NLIMBS - 1)}\f$.
   */
  template <unsigned NLIMBS>
  static DEVICE_INLINE void multiply_msb_raw_device(const storage<NLIMBS>& as, const storage<NLIMBS>& bs,  storage<2*NLIMBS>& rs)
  {
    if constexpr (NLIMBS > 1) {
      const uint32_t* a = as.limbs;
      const uint32_t* b = bs.limbs;
      uint32_t* even = rs.limbs;
      __align__(16) uint32_t odd[2 * NLIMBS - 2];

      even[NLIMBS - 1] = ptx::mul_hi(a[NLIMBS - 2], b[0]);
      odd[NLIMBS - 2] = ptx::mul_lo(a[NLIMBS - 1], b[0]);
      odd[NLIMBS - 1] = ptx::mul_hi(a[NLIMBS - 1], b[0]);
      size_t i;
      UNROLL
      for (i = 2; i < NLIMBS - 1; i += 2) {
        mad_row_msb<NLIMBS, true>(&even[NLIMBS - 2], &odd[NLIMBS - 2], &a[NLIMBS - i - 1], b[i - 1], i + 1);
        mad_row_msb<NLIMBS, false>(&odd[NLIMBS - 2], &even[NLIMBS - 2], &a[NLIMBS - i - 2], b[i], i + 2);
      }
      mad_row<NLIMBS>(&even[NLIMBS], &odd[NLIMBS - 2], a, b[NLIMBS - 1]);

      // merge |even| and |odd|
      ptx::add_cc(even[NLIMBS - 1], odd[NLIMBS - 2]);
      for (i = NLIMBS - 1; i < 2 * NLIMBS - 2; i++)
        even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
      even[i + 1] = ptx::addc(even[i + 1], 0);
    } else {
      multiply_raw_device<NLIMBS>(as, bs, rs);
    }
  }

  /**
   * A function that computes the low half of the fused multiply-and-add \f$ rs = as \cdot bs + cs \f$ where
   * \f$ bs = 2^{32*nof_limbs} \f$.
   *
   * For efficiency, this method does not include terms that are too large. Namely, limb product \f$ a_i \cdot b_j \f$
   * is excluded if \f$ i + j > NLIMBS - 1 \f$ and only the lower half is included if \f$ i + j = NLIMBS - 1 \f$. All other
   * limb products are included.
   */
  template <unsigned NLIMBS>
  static DEVICE_INLINE void
  multiply_and_add_lsb_neg_modulus_raw_device(const storage<NLIMBS>& as, const storage<NLIMBS>& bs, storage<NLIMBS>& cs, storage<NLIMBS>& rs)
  {
    const uint32_t* a = as.limbs;
    const uint32_t* b = bs.limbs;
    uint32_t* c = cs.limbs;
    uint32_t* even = rs.limbs;

    if constexpr (NLIMBS > 2) {
      __align__(16) uint32_t odd[NLIMBS - 1];
      size_t i;
      // `b[0]` is \f$ 2^{32} \f$ minus the last limb of prime modulus. Because most scalar (and some base) primes
      // are necessarily NTT-friendly, `b[0]` often turns out to be \f$ 2^{32} - 1 \f$. This actually leads to
      // less efficient SASS generated by nvcc, so this case needed separate handling.
      if (b[0] == UINT32_MAX) {
        add_sub_u32_device<NLIMBS, true, false>(c, a, even);
        for (i = 0; i < NLIMBS - 1; i++)
          odd[i] = a[i];
      } else {
        mul_n_and_add<NLIMBS>(even, a, b[0], c, NLIMBS);
        mul_n<NLIMBS>(odd, a + 1, b[0], NLIMBS - 1);
      }
      mad_row_lsb<NLIMBS>(&even[2], &odd[0], a, b[1], NLIMBS - 1);
      UNROLL
      for (i = 2; i < NLIMBS - 1; i += 2) {
        mad_row_lsb<NLIMBS>(&odd[i], &even[i], a, b[i], NLIMBS - i);
        mad_row_lsb<NLIMBS>(&even[i + 2], &odd[i], a, b[i + 1], NLIMBS - i - 1);
      }

      // merge |even| and |odd|
      even[1] = ptx::add_cc(even[1], odd[0]);
      for (i = 1; i < NLIMBS - 2; i++)
        even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
      even[i + 1] = ptx::addc(even[i + 1], odd[i]);
    } else if (NLIMBS == 2) {
      even[0] = ptx::mad_lo(a[0], b[0], c[0]);
      even[1] = ptx::mad_hi(a[0], b[0], c[0]);
      even[1] = ptx::mad_lo(a[0], b[1], even[1]);
      even[1] = ptx::mad_lo(a[1], b[0], even[1]);
    } else if (NLIMBS == 1) {
      even[0] = ptx::mad_lo(a[0], b[0], c[0]);
    }
  }

  
  // The following algorithms are adaptations of
  // http://www.acsel-lab.com/arithmetic/arith23/data/1616a047.pdf,
  // taken from https://github.com/z-prize/test-msm-gpu (under Apache 2.0 license)
  // and modified to use our datatypes.
  // We had our own implementation of http://www.acsel-lab.com/arithmetic/arith23/data/1616a047.pdf,
  // but the sppark versions achieved lower instruction count thanks to clever carry handling,
  // so we decided to just use theirs.
  template <unsigned NLIMBS>
  static DEVICE_INLINE void madc_n_rshift(uint32_t *odd, const uint32_t *a, uint32_t bi) {
#pragma unroll
    for (size_t i = 0; i < NLIMBS - 2; i += 2) {
      odd[i] = ptx::madc_lo_cc(a[i], bi, odd[i + 2]);
      odd[i + 1] = ptx::madc_hi_cc(a[i], bi, odd[i + 3]);
    }
    odd[NLIMBS - 2] = ptx::madc_lo_cc(a[NLIMBS - 2], bi, 0);
    odd[NLIMBS - 1] = ptx::madc_hi(a[NLIMBS - 2], bi, 0);
  }

  template <unsigned NLIMBS>
  static DEVICE_INLINE void mad_n_redc(uint32_t *even, uint32_t *odd, const uint32_t *a, uint32_t bi, const uint32_t *modulus, const uint32_t *mont_inv_modulus, bool first = false) {
    if (first) {
      mul_n<NLIMBS>(odd, a + 1, bi);
      mul_n<NLIMBS>(even, a, bi);
    } else {
      even[0] = ptx::add_cc(even[0], odd[1]);
      madc_n_rshift<NLIMBS>(odd, a + 1, bi);
      cmad_n<NLIMBS>(even, a, bi);
      odd[NLIMBS - 1] = ptx::addc(odd[NLIMBS - 1], 0);
    }
    uint32_t mi = even[0] * mont_inv_modulus[0];
    cmad_n<NLIMBS>(odd, modulus + 1, mi);
    cmad_n<NLIMBS>(even, modulus, mi);
    odd[NLIMBS - 1] = ptx::addc(odd[NLIMBS - 1], 0);
  }

  template <unsigned NLIMBS>
  static DEVICE_INLINE void qad_row(uint32_t *odd, uint32_t *even, const uint32_t *a, uint32_t bi, size_t n = NLIMBS) {
    cmad_n<NLIMBS>(odd, a, bi, n - 2);
    odd[n - 2] = ptx::madc_lo_cc(a[n - 2], bi, 0);
    odd[n - 1] = ptx::madc_hi(a[n - 2], bi, 0);
    cmad_n<NLIMBS>(even, a + 1, bi, n - 2);
    odd[n - 1] = ptx::addc(odd[n - 1], 0);
  }

  //TODO: test if beeter than karatsuba
  template <unsigned NLIMBS>
  static DEVICE_INLINE void multiply_raw_sb(const storage<NLIMBS> &as, const storage<NLIMBS> &bs, storage<2*NLIMBS> &rs) {
    const uint32_t *a = as.limbs;
    const uint32_t *b = bs.limbs;
    uint32_t *even = rs.limbs;
    __align__(8) uint32_t odd[2 * NLIMBS - 2];
    mul_n<NLIMBS>(even, a, b[0]);
    mul_n<NLIMBS>(odd, a + 1, b[0]);
    mad_row<NLIMBS>(&even[2], &odd[0], a, b[1]);
    size_t i;
#pragma unroll
    for (i = 2; i < NLIMBS - 1; i += 2) {
      mad_row<NLIMBS>(&odd[i], &even[i], a, b[i]);
      mad_row<NLIMBS>(&even[i + 2], &odd[i], a, b[i + 1]);
    }
    // merge |even| and |odd|
    even[1] = ptx::add_cc(even[1], odd[0]);
    for (i = 1; i < 2 * NLIMBS - 2; i++)
      even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
    even[i + 1] = ptx::addc(even[i + 1], 0);
  }

  template <unsigned NLIMBS>
  static DEVICE_INLINE void sqr_raw(const storage<NLIMBS> &as, storage<2*NLIMBS> &rs) {
    const uint32_t *a = as.limbs;
    uint32_t *even = rs.limbs;
    size_t i = 0, j;
    __align__(8) uint32_t odd[2 * NLIMBS - 2];

    // perform |a[i]|*|a[j]| for all j>i
    mul_n<NLIMBS>(even + 2, a + 2, a[0], NLIMBS - 2);
    mul_n<NLIMBS>(odd, a + 1, a[0], NLIMBS);

#pragma unroll
    while (i < NLIMBS - 4) {
      ++i;
      mad_row<NLIMBS>(&even[2 * i + 2], &odd[2 * i], &a[i + 1], a[i], NLIMBS - i - 1);
      ++i;
      qad_row<NLIMBS>(&odd[2 * i], &even[2 * i + 2], &a[i + 1], a[i], NLIMBS - i);
    }

    even[2 * NLIMBS - 4] = ptx::mul_lo(a[NLIMBS - 1], a[NLIMBS - 3]);
    even[2 * NLIMBS - 3] = ptx::mul_hi(a[NLIMBS - 1], a[NLIMBS - 3]);
    odd[2 * NLIMBS - 6] = ptx::mad_lo_cc(a[NLIMBS - 2], a[NLIMBS - 3], odd[2 * NLIMBS - 6]);
    odd[2 * NLIMBS - 5] = ptx::madc_hi_cc(a[NLIMBS - 2], a[NLIMBS - 3], odd[2 * NLIMBS - 5]);
    even[2 * NLIMBS - 3] = ptx::addc(even[2 * NLIMBS - 3], 0);

    odd[2 * NLIMBS - 4] = ptx::mul_lo(a[NLIMBS - 1], a[NLIMBS - 2]);
    odd[2 * NLIMBS - 3] = ptx::mul_hi(a[NLIMBS - 1], a[NLIMBS - 2]);

    // merge |even[2:]| and |odd[1:]|
    even[2] = ptx::add_cc(even[2], odd[1]);
    for (j = 2; j < 2 * NLIMBS - 3; j++)
      even[j + 1] = ptx::addc_cc(even[j + 1], odd[j]);
    even[j + 1] = ptx::addc(odd[j], 0);

    // double |even|
    even[0] = 0;
    even[1] = ptx::add_cc(odd[0], odd[0]);
    for (j = 2; j < 2 * NLIMBS - 1; j++)
      even[j] = ptx::addc_cc(even[j], even[j]);
    even[j] = ptx::addc(0, 0);

    // accumulate "diagonal" |a[i]|*|a[i]| product
    i = 0;
    even[2 * i] = ptx::mad_lo_cc(a[i], a[i], even[2 * i]);
    even[2 * i + 1] = ptx::madc_hi_cc(a[i], a[i], even[2 * i + 1]);
    for (++i; i < NLIMBS; i++) {
      even[2 * i] = ptx::madc_lo_cc(a[i], a[i], even[2 * i]);
      even[2 * i + 1] = ptx::madc_hi_cc(a[i], a[i], even[2 * i + 1]);
    }
  }

  template <unsigned NLIMBS>
  static DEVICE_INLINE void mul_by_1_row(uint32_t *even, uint32_t *odd, const uint32_t *modulus, const uint32_t *mont_inv_modulus, bool first = false) {
    uint32_t mi;
    if (first) {
      mi = even[0] * mont_inv_modulus[0];
      mul_n<NLIMBS>(odd, modulus + 1, mi);
      cmad_n<NLIMBS>(even, modulus, mi);
      odd[NLIMBS - 1] = ptx::addc(odd[NLIMBS - 1], 0);
    } else {
      even[0] = ptx::add_cc(even[0], odd[1]);
      // we trust the compiler to *not* touch the carry flag here
      // this code sits in between two "asm volatile" instructions witch should guarantee that nothing else interferes wit the carry flag
      mi = even[0] * mont_inv_modulus[0];
      madc_n_rshift<NLIMBS>(odd, modulus + 1, mi);
      cmad_n<NLIMBS>(even, modulus, mi);
      odd[NLIMBS - 1] = ptx::addc(odd[NLIMBS - 1], 0);
    }
  }

  // Performs Montgomery reduction on a storage<2*NLIMBS> input. Input value must be in the range [0, mod*2^(32*NLIMBS)).
  // Does not implement an in-place reduce<REDUCTION_SIZE> epilogue. If you want to further reduce the result,
  // call reduce<whatever>(xs.get_lo()) after the call to redc_wide_inplace.
  template <unsigned NLIMBS>
  static DEVICE_INLINE void reduce_mont_inplace(storage<2*NLIMBS> &xs, const storage<NLIMBS> &modulus, const storage<NLIMBS> &mont_inv_modulus) {
    uint32_t *even = xs.limbs;
    // Yields montmul of lo NLIMBS limbs * 1.
    // Since the hi NLIMBS limbs don't participate in computing the "mi" factor at each mul-and-rightshift stage,
    // it's ok to ignore the hi NLIMBS limbs during this process and just add them in afterward.
    uint32_t odd[NLIMBS];
    size_t i;
#pragma unroll
    for (i = 0; i < NLIMBS; i += 2) {
      mul_by_1_row<NLIMBS>(&even[0], &odd[0], modulus.limbs, mont_inv_modulus.limbs, i == 0);
      mul_by_1_row<NLIMBS>(&odd[0], &even[0], modulus.limbs, mont_inv_modulus.limbs);
    }
    even[0] = ptx::add_cc(even[0], odd[1]);
#pragma unroll
    for (i = 1; i < NLIMBS - 1; i++)
      even[i] = ptx::addc_cc(even[i], odd[i + 1]);
    even[i] = ptx::addc(even[i], 0);
    // Adds in (hi NLIMBS limbs), implicitly right-shifting them by NLIMBS limbs as if they had participated in the
    // add-and-rightshift stages above.
    xs.limbs[0] = ptx::add_cc(xs.limbs[0], xs.limbs[NLIMBS]);
#pragma unroll
    for (i = 1; i < NLIMBS - 1; i++)
      xs.limbs[i] = ptx::addc_cc(xs.limbs[i], xs.limbs[i + NLIMBS]);
    xs.limbs[NLIMBS - 1] = ptx::addc(xs.limbs[NLIMBS - 1], xs.limbs[2 * NLIMBS - 1]);
  }

  template <unsigned NLIMBS>
  static DEVICE_INLINE void montmul_raw(const storage<NLIMBS> &a_in, const storage<NLIMBS> &b_in, const storage<NLIMBS> &modulus, const storage<NLIMBS> &mont_inv_modulus, storage<NLIMBS> &r_in) {
    const uint32_t *a = a_in.limbs;
    const uint32_t *b = b_in.limbs;
    uint32_t *even = r_in.limbs;
    __align__(8) uint32_t odd[NLIMBS + 1];
    size_t i;
#pragma unroll
    for (i = 0; i < NLIMBS; i += 2) {
      mad_n_redc<NLIMBS>(&even[0], &odd[0], a, b[i], modulus.limbs, mont_inv_modulus.limbs, i == 0);
      mad_n_redc<NLIMBS>(&odd[0], &even[0], a, b[i + 1], modulus.limbs, mont_inv_modulus.limbs);
    }
    // merge |even| and |odd|
    even[0] = ptx::add_cc(even[0], odd[1]);
#pragma unroll
    for (i = 1; i < NLIMBS - 1; i++)
      even[i] = ptx::addc_cc(even[i], odd[i + 1]);
    even[i] = ptx::addc(even[i], 0);
    // final reduction from [0, 2*mod) to [0, mod) not done here, instead performed optionally in mul_device wrapper
  }


  // Device path adapts http://www.acsel-lab.com/arithmetic/arith23/data/1616a047.pdf to use IMAD.WIDE.
  template <unsigned NLIMBS> static constexpr DEVICE_INLINE storage<NLIMBS> mulmont_device(const storage<NLIMBS> &xs, const storage<NLIMBS> &ys, const storage<NLIMBS> &modulus, const storage<NLIMBS> &mont_inv_modulus) {
    // Forces us to think more carefully about the last carry bit if we use a modulus with fewer than 2 leading zeroes of slack
    // static_assert(!(CONFIG::modulus.limbs[NLIMBS - 1] >> 30));
    // printf(" ");
    storage<NLIMBS> rs = {0};
    montmul_raw(xs, ys, modulus, mont_inv_modulus, rs);
    return rs;
  }

  template <unsigned NLIMBS> static constexpr DEVICE_INLINE storage<NLIMBS> sqrmont_device(const storage<NLIMBS> &xs) {
    // Forces us to think more carefully about the last carry bit if we use a modulus with fewer than 2 leading zeroes of slack
    // static_assert(!(CONFIG::modulus.limbs[NLIMBS - 1] >> 30));
    storage<2*NLIMBS> rs = {0};
    sqr_raw(xs, rs);
    redc_wide_inplace(rs); // after reduce_twopass, tmp's low NLIMBS limbs should represent a value in [0, 2*mod)
    return rs.get_lo();
  }
// //add
//   // return xs * ys with field operands
//   // Device path adapts http://www.acsel-lab.com/arithmetic/arith23/data/1616a047.pdf to use IMAD.WIDE.
//   // Host path uses CIOS.
//   template <unsigned REDUCTION_SIZE = 1> static constexpr DEVICE_INLINE storage<NLIMBS> mulz(const storage<NLIMBS> &xs, const storage<NLIMBS> &ys) {
//     return mul_devicez<REDUCTION_SIZE>(xs, ys);
//   }
}

#endif