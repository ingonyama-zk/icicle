#pragma once

#include "../utils/host_math.cuh"
#include "../utils/ptx.cuh"
#include "../utils/storage.cuh"
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#define HOST_INLINE        __host__ __forceinline__
#define DEVICE_INLINE      __device__ __forceinline__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__

template <class CONFIG>
class Field
{
public:
  static constexpr unsigned TLC = CONFIG::limbs_count;
  static constexpr unsigned NBITS = CONFIG::modulus_bit_count;

  static constexpr HOST_DEVICE_INLINE Field zero() { return Field{CONFIG::zero}; }

  static constexpr HOST_DEVICE_INLINE Field one() { return Field{CONFIG::one}; }

  static constexpr HOST_DEVICE_INLINE Field from(uint32_t value)
  {
    storage<TLC> scalar;
    scalar.limbs[0] = value;
    for (int i = 1; i < TLC; i++) {
      scalar.limbs[i] = 0;
    }
    return Field{scalar};
  }

  static HOST_INLINE Field omega(uint32_t logn)
  {
    if (logn == 0) { return Field{CONFIG::one}; }

    if (logn > CONFIG::omegas_count) { throw std::invalid_argument("Field: Invalid omega index"); }

    storage_array<CONFIG::omegas_count, TLC> const omega = CONFIG::omega;
    return Field{omega.storages[logn - 1]};
  }

  static HOST_INLINE Field omega_inv(uint32_t logn)
  {
    if (logn == 0) { return Field{CONFIG::one}; }

    if (logn > CONFIG::omegas_count) { throw std::invalid_argument("Field: Invalid omega_inv index"); }

    storage_array<CONFIG::omegas_count, TLC> const omega_inv = CONFIG::omega_inv;
    return Field{omega_inv.storages[logn - 1]};
  }

  static HOST_INLINE Field inv_log_size(uint32_t logn)
  {
    if (logn == 0) { return Field{CONFIG::one}; }

    if (logn > CONFIG::omegas_count) { throw std::invalid_argument("Field: Invalid inv index"); }
    storage_array<CONFIG::omegas_count, TLC> const inv = CONFIG::inv;
    return Field{inv.storages[logn - 1]};
  }

  // private:
  typedef storage<TLC> ff_storage;
  typedef storage<2 * TLC> ff_wide_storage;

  static constexpr HOST_DEVICE_INLINE ff_storage get_neg_modulus() { return CONFIG::neg_modulus; }

  static constexpr HOST_DEVICE_INLINE unsigned num_of_reductions() { return CONFIG::num_of_reductions; }

  static constexpr unsigned slack_bits = 32 * TLC - NBITS;

  struct Wide {
    ff_wide_storage limbs_storage;

    static constexpr Field HOST_DEVICE_INLINE get_lower(const Wide& xs)
    {
      Field out{};
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
      for (unsigned i = 0; i < TLC; i++)
        out.limbs_storage.limbs[i] = xs.limbs_storage.limbs[i];
      return out;
    }

    static constexpr Field HOST_DEVICE_INLINE get_higher(const Wide& xs)
    {
      Field out{};
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
      for (unsigned i = 0; i < TLC; i++)
        out.limbs_storage.limbs[i] = xs.limbs_storage.limbs[i + TLC];
      return out;
    }

    static constexpr Field HOST_DEVICE_INLINE get_higher_with_slack(const Wide& xs)
    {
      Field out{};
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
      for (unsigned i = 0; i < TLC; i++) {
#ifdef __CUDA_ARCH__
        out.limbs_storage.limbs[i] =
          __funnelshift_lc(xs.limbs_storage.limbs[i + TLC - 1], xs.limbs_storage.limbs[i + TLC], 2 * slack_bits);
#else
        out.limbs_storage.limbs[i] = (xs.limbs_storage.limbs[i + TLC] << 2 * slack_bits) +
                                     (xs.limbs_storage.limbs[i + TLC - 1] >> (32 - 2 * slack_bits));
#endif
      }
      return out;
    }

    template <unsigned REDUCTION_SIZE = 1>
    static constexpr HOST_DEVICE_INLINE Wide sub_modulus_squared(const Wide& xs)
    {
      if (REDUCTION_SIZE == 0) return xs;
      const ff_wide_storage modulus = get_modulus_squared<REDUCTION_SIZE>();
      Wide rs = {};
      return sub_limbs<true>(xs.limbs_storage, modulus, rs.limbs_storage) ? xs : rs;
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE Wide neg(const Wide& xs)
    {
      const ff_wide_storage modulus = get_modulus_squared<MODULUS_MULTIPLE>();
      Wide rs = {};
      sub_limbs<false>(modulus, xs.limbs_storage, rs.limbs_storage);
      return rs;
    }

    friend HOST_DEVICE_INLINE Wide operator+(Wide xs, const Wide& ys)
    {
      Wide rs = {};
      add_limbs<false>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
      return sub_modulus_squared<1>(rs);
    }

    friend HOST_DEVICE_INLINE Wide operator-(Wide xs, const Wide& ys)
    {
      Wide rs = {};
      uint32_t carry = sub_limbs<true>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
      if (carry == 0) return rs;
      const ff_wide_storage modulus = get_modulus_squared<1>();
      add_limbs<false>(rs.limbs_storage, modulus, rs.limbs_storage);
      return rs;
    }
  };

  // return modulus
  template <unsigned MULTIPLIER = 1>
  static constexpr HOST_DEVICE_INLINE ff_storage get_modulus()
  {
    switch (MULTIPLIER) {
    case 1:
      return CONFIG::modulus;
    case 2:
      return CONFIG::modulus_2;
    case 4:
      return CONFIG::modulus_4;
    default:
      return {};
    }
  }

  template <unsigned MULTIPLIER = 1>
  static constexpr HOST_DEVICE_INLINE ff_wide_storage modulus_wide()
  {
    return CONFIG::modulus_wide;
  }

  // return m
  static constexpr HOST_DEVICE_INLINE ff_storage get_m() { return CONFIG::m; }

  // return modulus^2, helpful for ab +/- cd
  template <unsigned MULTIPLIER = 1>
  static constexpr HOST_DEVICE_INLINE ff_wide_storage get_modulus_squared()
  {
    switch (MULTIPLIER) {
    case 1:
      return CONFIG::modulus_squared;
    case 2:
      return CONFIG::modulus_squared_2;
    case 4:
      return CONFIG::modulus_squared_4;
    default:
      return {};
    }
  }

  template <bool SUBTRACT, bool CARRY_OUT>
  static constexpr __device__ __forceinline__ uint32_t
  add_sub_u32_device(const uint32_t* x, const uint32_t* y, uint32_t* r, size_t n = (TLC >> 1))
  {
    r[0] = SUBTRACT ? ptx::sub_cc(x[0], y[0]) : ptx::add_cc(x[0], y[0]);
    for (unsigned i = 1; i < (CARRY_OUT ? n : n - 1); i++)
      r[i] = SUBTRACT ? ptx::subc_cc(x[i], y[i]) : ptx::addc_cc(x[i], y[i]);
    if (!CARRY_OUT) {
      r[n - 1] = SUBTRACT ? ptx::subc(x[n - 1], y[n - 1]) : ptx::addc(x[n - 1], y[n - 1]);
      return 0;
    }
    return SUBTRACT ? ptx::subc(0, 0) : ptx::addc(0, 0);
  }

  // add or subtract limbs
  template <bool SUBTRACT, bool CARRY_OUT>
  static constexpr DEVICE_INLINE uint32_t
  add_sub_limbs_device(const ff_storage& xs, const ff_storage& ys, ff_storage& rs)
  {
    const uint32_t* x = xs.limbs;
    const uint32_t* y = ys.limbs;
    uint32_t* r = rs.limbs;
    return add_sub_u32_device<SUBTRACT, CARRY_OUT>(x, y, r, TLC);
  }

  template <bool SUBTRACT, bool CARRY_OUT>
  static constexpr DEVICE_INLINE uint32_t
  add_sub_limbs_device(const ff_wide_storage& xs, const ff_wide_storage& ys, ff_wide_storage& rs)
  {
    const uint32_t* x = xs.limbs;
    const uint32_t* y = ys.limbs;
    uint32_t* r = rs.limbs;
    return add_sub_u32_device<SUBTRACT, CARRY_OUT>(x, y, r, 2 * TLC);
  }

  template <bool SUBTRACT, bool CARRY_OUT>
  static constexpr HOST_INLINE uint32_t add_sub_limbs_host(const ff_storage& xs, const ff_storage& ys, ff_storage& rs)
  {
    const uint32_t* x = xs.limbs;
    const uint32_t* y = ys.limbs;
    uint32_t* r = rs.limbs;
    uint32_t carry = 0;
    host_math::carry_chain<TLC, false, CARRY_OUT> chain;
    for (unsigned i = 0; i < TLC; i++)
      r[i] = SUBTRACT ? chain.sub(x[i], y[i], carry) : chain.add(x[i], y[i], carry);
    return CARRY_OUT ? carry : 0;
  }

  template <bool SUBTRACT, bool CARRY_OUT>
  static constexpr HOST_INLINE uint32_t
  add_sub_limbs_host(const ff_wide_storage& xs, const ff_wide_storage& ys, ff_wide_storage& rs)
  {
    const uint32_t* x = xs.limbs;
    const uint32_t* y = ys.limbs;
    uint32_t* r = rs.limbs;
    uint32_t carry = 0;
    host_math::carry_chain<2 * TLC, false, CARRY_OUT> chain;
    for (unsigned i = 0; i < 2 * TLC; i++)
      r[i] = SUBTRACT ? chain.sub(x[i], y[i], carry) : chain.add(x[i], y[i], carry);
    return CARRY_OUT ? carry : 0;
  }

  template <bool CARRY_OUT, typename T>
  static constexpr HOST_DEVICE_INLINE uint32_t add_limbs(const T& xs, const T& ys, T& rs)
  {
#ifdef __CUDA_ARCH__
    return add_sub_limbs_device<false, CARRY_OUT>(xs, ys, rs);
#else
    return add_sub_limbs_host<false, CARRY_OUT>(xs, ys, rs);
#endif
  }

  template <bool CARRY_OUT, typename T>
  static constexpr HOST_DEVICE_INLINE uint32_t sub_limbs(const T& xs, const T& ys, T& rs)
  {
#ifdef __CUDA_ARCH__
    return add_sub_limbs_device<true, CARRY_OUT>(xs, ys, rs);
#else
    return add_sub_limbs_host<true, CARRY_OUT>(xs, ys, rs);
#endif
  }

  static DEVICE_INLINE void mul_n(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = TLC)
  {
#pragma unroll
    for (size_t i = 0; i < n; i += 2) {
      acc[i] = ptx::mul_lo(a[i], bi);
      acc[i + 1] = ptx::mul_hi(a[i], bi);
    }
  }

  static DEVICE_INLINE void mul_n_msb(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = TLC, size_t start_i = 0)
  {
#pragma unroll
    for (size_t i = start_i; i < n; i += 2) {
      acc[i] = ptx::mul_lo(a[i], bi);
      acc[i + 1] = ptx::mul_hi(a[i], bi);
    }
  }

  template <bool CARRY_IN = false>
  static __device__ __forceinline__ void
  cmad_n(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = TLC, uint32_t optional_carry = 0)
  {
    if (CARRY_IN) ptx::add_cc(UINT32_MAX, optional_carry);
    acc[0] = CARRY_IN ? ptx::madc_lo_cc(a[0], bi, acc[0]) : ptx::mad_lo_cc(a[0], bi, acc[0]);
    acc[1] = ptx::madc_hi_cc(a[0], bi, acc[1]);

#pragma unroll
    for (size_t i = 2; i < n; i += 2) {
      acc[i] = ptx::madc_lo_cc(a[i], bi, acc[i]);
      acc[i + 1] = ptx::madc_hi_cc(a[i], bi, acc[i + 1]);
    }
  }

  template <bool EVEN_PHASE>
  static __device__ __forceinline__ void cmad_n_msb(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = TLC)
  {
    if (EVEN_PHASE) {
      acc[0] = ptx::mad_lo_cc(a[0], bi, acc[0]);
      acc[1] = ptx::madc_hi_cc(a[0], bi, acc[1]);
    } else {
      acc[1] = ptx::mad_hi_cc(a[0], bi, acc[1]);
    }

#pragma unroll
    for (size_t i = 2; i < n; i += 2) {
      acc[i] = ptx::madc_lo_cc(a[i], bi, acc[i]);
      acc[i + 1] = ptx::madc_hi_cc(a[i], bi, acc[i + 1]);
    }
  }

  static __device__ __forceinline__ void cmad_n_lsb(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = TLC)
  {
    if (n > 1)
      acc[0] = ptx::mad_lo_cc(a[0], bi, acc[0]);
    else
      acc[0] = ptx::mad_lo(a[0], bi, acc[0]);

    size_t i;
#pragma unroll
    for (i = 1; i < n - 1; i += 2) {
      acc[i] = ptx::madc_hi_cc(a[i - 1], bi, acc[i]);
      if (i == n - 2)
        acc[i + 1] = ptx::madc_lo(a[i + 1], bi, acc[i + 1]);
      else
        acc[i + 1] = ptx::madc_lo_cc(a[i + 1], bi, acc[i + 1]);
    }
    if (i == n - 1) acc[i] = ptx::madc_hi(a[i - 1], bi, acc[i]);
  }

  template <bool CARRY_OUT = false, bool CARRY_IN = false>
  static __device__ __forceinline__ uint32_t mad_row(
    uint32_t* odd,
    uint32_t* even,
    const uint32_t* a,
    uint32_t bi,
    size_t n = TLC,
    uint32_t ci = 0,
    uint32_t di = 0,
    uint32_t carry_for_high = 0,
    uint32_t carry_for_low = 0)
  {
    cmad_n<CARRY_IN>(odd, a + 1, bi, n - 2, carry_for_low);
    odd[n - 2] = ptx::madc_lo_cc(a[n - 1], bi, ci);
    odd[n - 1] = CARRY_OUT ? ptx::madc_hi_cc(a[n - 1], bi, di) : ptx::madc_hi(a[n - 1], bi, di);
    uint32_t cr = CARRY_OUT ? ptx::addc(0, 0) : 0;
    cmad_n(even, a, bi, n);
    if (CARRY_OUT) { 
      odd[n - 1] = ptx::addc_cc(odd[n - 1], carry_for_high);
      cr = ptx::addc(cr, 0);
    } else
      odd[n - 1] = ptx::addc(odd[n - 1], carry_for_high);
    return cr;
  }

  template <bool EVEN_PHASE>
  static __device__ __forceinline__ void
  mad_row_msb(uint32_t* odd, uint32_t* even, const uint32_t* a, uint32_t bi, size_t n = TLC)
  {
    cmad_n_msb<!EVEN_PHASE>(odd, EVEN_PHASE ? a : (a + 1), bi, n - 2);
    odd[EVEN_PHASE ? (n - 1) : (n - 2)] = ptx::madc_lo_cc(a[n - 1], bi, 0);
    odd[EVEN_PHASE ? n : (n - 1)] = ptx::madc_hi(a[n - 1], bi, 0);
    cmad_n_msb<EVEN_PHASE>(even, EVEN_PHASE ? (a + 1) : a, bi, n - 1);
    odd[EVEN_PHASE ? n : (n - 1)] = ptx::addc(odd[EVEN_PHASE ? n : (n - 1)], 0);
  }

  static __device__ __forceinline__ void
  mad_row_lsb(uint32_t* odd, uint32_t* even, const uint32_t* a, uint32_t bi, size_t n = TLC)
  {
    if (bi != 0) {
        if (n > 1) cmad_n_lsb(odd, a + 1, bi, n - 1);
        cmad_n_lsb(even, a, bi, n);
    }
    return;
  }

  static __device__ __forceinline__ uint32_t
  mul_n_plus_extra(uint32_t* acc, const uint32_t* a, uint32_t bi, uint32_t* extra, size_t n = (TLC >> 1))
  {
    acc[0] = ptx::mad_lo_cc(a[0], bi, extra[0]);

#pragma unroll
    for (size_t i = 1; i < n - 1; i += 2) {
      acc[i] = ptx::madc_hi_cc(a[i - 1], bi, extra[i]);
      acc[i + 1] = ptx::madc_lo_cc(a[i + 1], bi, extra[i + 1]);
    }

    acc[n - 1] = ptx::madc_hi_cc(a[n - 2], bi, extra[n - 1]);
    return ptx::addc(0, 0);
  }

  static DEVICE_INLINE void mult_no_carry(uint32_t a, uint32_t b, uint32_t* r)
  {
    r[0] = ptx::mul_lo(a, b);
    r[1] = ptx::mul_hi(a, b);
  }

  static __device__ __forceinline__ void
  multiply_msb_raw_device(const ff_storage& as, const ff_storage& bs, ff_wide_storage& rs)
  {
    // r = a * b is almost correct for the higher TLC + 1 digits
    const uint32_t* a = as.limbs;
    const uint32_t* b = bs.limbs;
    uint32_t* even = rs.limbs;
    __align__(16) uint32_t odd[2 * TLC - 2];

    even[TLC - 1] = ptx::mul_hi(a[TLC - 2], b[0]);
    odd[TLC - 2] = ptx::mul_lo(a[TLC - 1], b[0]);
    odd[TLC - 1] = ptx::mul_hi(a[TLC - 1], b[0]);
    size_t i;
#pragma unroll
    for (i = 2; i < TLC - 1; i += 2) {
      mad_row_msb<true>(&even[TLC - 2], &odd[TLC - 2], &a[TLC - i - 1], b[i - 1], i + 1);
      mad_row_msb<false>(&odd[TLC - 2], &even[TLC - 2], &a[TLC - i - 2], b[i], i + 2);
    }
    mad_row(&even[TLC], &odd[TLC - 2], a, b[TLC - 1]);

    // merge |even| and |odd|
    ptx::add_cc(even[TLC - 1], odd[TLC - 2]);
    for (i = TLC - 1; i < 2 * TLC - 2; i++)
      even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
    even[i + 1] = ptx::addc(even[i + 1], 0);
  }

  static __device__ __forceinline__ void
  multiply_and_add_lsb_raw_device(const ff_storage& as, const ff_storage& bs, ff_storage& cs, ff_storage& rs)
  {
    // r = a * b + c is correct for the lower TLC digits
    const uint32_t* a = as.limbs;
    const uint32_t* b = bs.limbs;
    uint32_t* even = rs.limbs;
    __align__(16) uint32_t odd[TLC - 1];
    size_t i;
    if (b[0] == UINT32_MAX) {
      add_sub_u32_device<true, false>(cs.limbs, a, even, TLC);
      for (i = 0; i < TLC - 1; i++)
        odd[i] = a[i];
    } else {
      mul_n_plus_extra(even, a, b[0], cs.limbs, TLC);
      mul_n(odd, a + 1, b[0], TLC - 1);
    }
    mad_row_lsb(&even[2], &odd[0], a, b[1], TLC - 1);
#pragma unroll
    for (i = 2; i < TLC - 1; i += 2) {
      mad_row_lsb(&odd[i], &even[i], a, b[i], TLC - i);
      mad_row_lsb(&even[i + 2], &odd[i], a, b[i + 1], TLC - i - 1);
    }

    // merge |even| and |odd|
    even[1] = ptx::add_cc(even[1], odd[0]);
    for (i = 1; i < TLC - 2; i++)
      even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
    even[i + 1] = ptx::addc(even[i + 1], odd[i]);
  }

  // This method multiplies `a` and `b` and adds `in1` and `in2` to the result
  // It is used to compute the "middle" part of Karatsuba: `a0 * b1 + b0 * a1`
  // So under the assumption that the top bits of `a` and `b` are unset, we can ignore all the carries from here
  static __device__ __forceinline__ void
  multiply_and_add_short_raw_device(const uint32_t* a, const uint32_t* b, uint32_t* even, uint32_t* in1, uint32_t* in2)
  {
    __align__(16) uint32_t odd[TLC - 2];
    uint32_t first_row_carry = mul_n_plus_extra(even, a, b[0], in1);
    uint32_t carry = mul_n_plus_extra(odd, a + 1, b[0], &in2[1]);

    size_t i;
#pragma unroll
    for (i = 2; i < ((TLC >> 1) - 1); i += 2) {
      carry = mad_row<true, false>(
        &even[i], &odd[i - 2], a, b[i - 1], TLC >> 1, in1[(TLC >> 1) + i - 2], in1[(TLC >> 1) + i - 1], carry);
      carry =
        mad_row<true, false>(&odd[i], &even[i], a, b[i], TLC >> 1, in2[(TLC >> 1) + i - 1], in2[(TLC >> 1) + i], carry);
    }
    mad_row<false, true>(
      &even[TLC >> 1], &odd[(TLC >> 1) - 2], a, b[(TLC >> 1) - 1], TLC >> 1, in1[TLC - 2], in1[TLC - 1], carry,
      first_row_carry);
    // merge |even| and |odd| plus the parts of in2 we haven't added yet
    even[0] = ptx::add_cc(even[0], in2[0]);
    for (i = 0; i < (TLC - 2); i++)
      even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
    even[i + 1] = ptx::addc(even[i + 1], in2[i + 1]);
  }

  static __device__ __forceinline__ void multiply_short_raw_device(const uint32_t* a, const uint32_t* b, uint32_t* even)
  {
    __align__(16) uint32_t odd[TLC - 2];
    mul_n(even, a, b[0], TLC >> 1);
    mul_n(odd, a + 1, b[0], TLC >> 1);
    mad_row(&even[2], &odd[0], a, b[1], TLC >> 1);

    size_t i;
#pragma unroll
    for (i = 2; i < ((TLC >> 1) - 1); i += 2) {
      mad_row(&odd[i], &even[i], a, b[i], TLC >> 1);
      mad_row(&even[i + 2], &odd[i], a, b[i + 1], TLC >> 1);
    }
    // merge |even| and |odd|
    even[1] = ptx::add_cc(even[1], odd[0]);
    for (i = 1; i < TLC - 2; i++)
      even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
    even[i + 1] = ptx::addc(even[i + 1], 0);
  }

  static DEVICE_INLINE void multiply_raw_device(const ff_storage& as, const ff_storage& bs, ff_wide_storage& rs)
  {
    const uint32_t* a = as.limbs;
    const uint32_t* b = bs.limbs;
    uint32_t* r = rs.limbs;
    multiply_short_raw_device(a, b, r);
    multiply_short_raw_device(&a[TLC >> 1], &b[TLC >> 1], &r[TLC]);
    __align__(16) uint32_t middle_part[TLC];
    __align__(16) uint32_t diffs[TLC];
    uint32_t carry1 = add_sub_u32_device<true, true>(&a[TLC >> 1], a, diffs);
    uint32_t carry2 = add_sub_u32_device<true, true>(b, &b[TLC >> 1], &diffs[TLC >> 1]);
    multiply_and_add_short_raw_device(diffs, &diffs[TLC >> 1], middle_part, r, &r[TLC]);
    if (carry1) add_sub_u32_device<true, false>(&middle_part[TLC >> 1], &diffs[TLC >> 1], &middle_part[TLC >> 1]);
    if (carry2) add_sub_u32_device<true, false>(&middle_part[TLC >> 1], diffs, &middle_part[TLC >> 1]);
    add_sub_u32_device<false, true>(&r[TLC >> 1], middle_part, &r[TLC >> 1], TLC);

    for (size_t i = TLC + (TLC >> 1); i < 2 * TLC; i++)
      r[i] = ptx::addc_cc(r[i], 0);
  }

  static HOST_INLINE void multiply_raw_host(const ff_storage& as, const ff_storage& bs, ff_wide_storage& rs)
  {
    const uint32_t* a = as.limbs;
    const uint32_t* b = bs.limbs;
    uint32_t* r = rs.limbs;
    for (unsigned i = 0; i < TLC; i++) {
      uint32_t carry = 0;
      for (unsigned j = 0; j < TLC; j++)
        r[j + i] = host_math::madc_cc(a[j], b[i], r[j + i], carry);
      r[TLC + i] = carry;
    }
  }

  static HOST_DEVICE_INLINE void multiply_raw(const ff_storage& as, const ff_storage& bs, ff_wide_storage& rs)
  {
#ifdef __CUDA_ARCH__
    return multiply_raw_device(as, bs, rs);
#else
    return multiply_raw_host(as, bs, rs);
#endif
  }

  static HOST_DEVICE_INLINE void multiply_and_add_lsb_raw(const ff_storage& as, const ff_storage& bs, ff_storage& cs, ff_storage& rs)
  {
#ifdef __CUDA_ARCH__
    return multiply_and_add_lsb_raw_device(as, bs, cs, rs);
#else
    Wide r_wide = {};
    multiply_raw_host(as, bs, r_wide.limbs_storage);
    Field r = Wide::get_lower(r_wide);
    add_limbs<false>(cs, r.limbs_storage, rs);
#endif
  }

  static HOST_DEVICE_INLINE void multiply_msb_raw(const ff_storage& as, const ff_storage& bs, ff_wide_storage& rs)
  {
#ifdef __CUDA_ARCH__
    return multiply_msb_raw_device(as, bs, rs);
#else
    return multiply_raw_host(as, bs, rs);
#endif
  }

public:
  ff_storage limbs_storage;

  HOST_DEVICE_INLINE uint32_t* export_limbs() { return (uint32_t*)limbs_storage.limbs; }

  HOST_DEVICE_INLINE unsigned get_scalar_digit(unsigned digit_num, unsigned digit_width)
  {
    const uint32_t limb_lsb_idx = (digit_num * digit_width) / 32;
    const uint32_t shift_bits = (digit_num * digit_width) % 32;
    unsigned rv = limbs_storage.limbs[limb_lsb_idx] >> shift_bits;
    if ((shift_bits + digit_width > 32) && (limb_lsb_idx + 1 < TLC)) {
      rv += limbs_storage.limbs[limb_lsb_idx + 1] << (32 - shift_bits);
    }
    rv &= ((1 << digit_width) - 1);
    return rv;
  }

  static HOST_INLINE Field rand_host()
  {
    std::random_device rd;
    std::mt19937_64 generator(rd());
    std::uniform_int_distribution<unsigned> distribution;
    Field value{};
    for (unsigned i = 0; i < TLC; i++)
      value.limbs_storage.limbs[i] = distribution(generator);
    while (lt(Field { get_modulus() }, value))
      value = value - Field { get_modulus() };
    return value;
  }

  template <unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE Field sub_modulus(const Field& xs)
  {
    if (REDUCTION_SIZE == 0) return xs;
    const ff_storage modulus = get_modulus<REDUCTION_SIZE>();
    Field rs = {};
    return sub_limbs<true>(xs.limbs_storage, modulus, rs.limbs_storage) ? xs : rs;
  }

  friend std::ostream& operator<<(std::ostream& os, const Field& xs)
  {
    std::stringstream hex_string;
    hex_string << std::hex << std::setfill('0');

    for (int i = 0; i < TLC; i++) {
      hex_string << std::setw(8) << xs.limbs_storage.limbs[TLC - i - 1];
    }

    os << "0x" << hex_string.str();
    return os;
  }

  friend HOST_DEVICE_INLINE Field operator+(Field xs, const Field& ys)
  {
    Field rs = {};
    add_limbs<false>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
    return sub_modulus<1>(rs);
  }

  friend HOST_DEVICE_INLINE Field operator-(Field xs, const Field& ys)
  {
    Field rs = {};
    uint32_t carry = sub_limbs<true>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
    if (carry == 0) return rs;
    const ff_storage modulus = get_modulus<1>();
    add_limbs<false>(rs.limbs_storage, modulus, rs.limbs_storage);
    return rs;
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Wide mul_wide(const Field& xs, const Field& ys)
  {
    Wide rs = {};
    multiply_raw(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
    return rs;
  }

  static constexpr HOST_DEVICE_INLINE Field to_montgomery(const Field& xs) { return xs * Field{CONFIG::montgomery_r}; }

  static constexpr HOST_DEVICE_INLINE Field from_montgomery(const Field& xs)
  {
    return xs * Field{CONFIG::montgomery_r_inv};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Field reduce(const Wide& xs)
  {
    Field xs_hi = Wide::get_higher_with_slack(xs); // xy << slack_bits
    Wide l = {};
    multiply_msb_raw(xs_hi.limbs_storage, get_m(), l.limbs_storage); // MSB mult
    Field l_hi = Wide::get_higher(l);
    Field r = {};
    Field xs_lo = Wide::get_lower(xs);
    multiply_and_add_lsb_raw(l_hi.limbs_storage, get_neg_modulus(), xs_lo.limbs_storage, r.limbs_storage); // LSB mad
    ff_storage r_reduced = {};
    uint32_t carry;
    if (num_of_reductions() == 2) {
      carry = sub_limbs<true>(r.limbs_storage, get_modulus<2>(), r_reduced);
      if (carry == 0)
        r = Field { r_reduced };
    }
    carry = sub_limbs<true>(r.limbs_storage, get_modulus<1>(), r_reduced);
    if (carry == 0)
      r = Field { r_reduced };

    return r;
  }

  friend HOST_DEVICE_INLINE Field operator*(const Field& xs, const Field& ys)
  {
    Wide xy = mul_wide(xs, ys); // full mult
    return reduce(xy);
  }

  friend HOST_DEVICE_INLINE bool operator==(const Field& xs, const Field& ys)
  {
#ifdef __CUDA_ARCH__
    const uint32_t* x = xs.limbs_storage.limbs;
    const uint32_t* y = ys.limbs_storage.limbs;
    uint32_t limbs_or = x[0] ^ y[0];
#pragma unroll
    for (unsigned i = 1; i < TLC; i++)
      limbs_or |= x[i] ^ y[i];
    return limbs_or == 0;
#else
    for (unsigned i = 0; i < TLC; i++)
      if (xs.limbs_storage.limbs[i] != ys.limbs_storage.limbs[i]) return false;
    return true;
#endif
  }

  friend HOST_DEVICE_INLINE bool operator!=(const Field& xs, const Field& ys) { return !(xs == ys); }

  template <const Field& multiplier>
  static HOST_DEVICE_INLINE Field mul_const(const Field& xs)
  {
    Field mul = multiplier;
    static bool is_u32 = true;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (unsigned i = 1; i < TLC; i++)
      is_u32 &= (mul.limbs_storage.limbs[i] == 0);

    if (is_u32) return mul_unsigned<multiplier.limbs_storage.limbs[0], Field>(xs);
    return mul * xs;
  }

  template <uint32_t mutliplier, class T, unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE T mul_unsigned(const T& xs)
  {
    T rs = {};
    T temp = xs;
    bool is_zero = true;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (unsigned i = 0; i < 32; i++) {
      if (mutliplier & (1 << i)) {
        rs = is_zero ? temp : (rs + temp);
        is_zero = false;
      }
      if (mutliplier & ((1 << (31 - i) - 1) << (i + 1))) break;
      temp = temp + temp;
    }
    return rs;
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Wide sqr_wide(const Field& xs)
  {
    // TODO: change to a more efficient squaring
    return mul_wide<MODULUS_MULTIPLE>(xs, xs);
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Field sqr(const Field& xs)
  {
    // TODO: change to a more efficient squaring
    return xs * xs;
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Field neg(const Field& xs)
  {
    const ff_storage modulus = get_modulus<MODULUS_MULTIPLE>();
    Field rs = {};
    sub_limbs<false>(modulus, xs.limbs_storage, rs.limbs_storage);
    return rs;
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Field div2(const Field& xs)
  {
    const uint32_t* x = xs.limbs_storage.limbs;
    Field rs = {};
    uint32_t* r = rs.limbs_storage.limbs;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (unsigned i = 0; i < TLC - 1; i++) {
#ifdef __CUDA_ARCH__
      r[i] = __funnelshift_rc(x[i], x[i + 1], 1);
#else
      r[i] = (x[i] >> 1) | (x[i + 1] << 31);
#endif
    }
    r[TLC - 1] = x[TLC - 1] >> 1;
    return sub_modulus<MODULUS_MULTIPLE>(rs);
  }

  static constexpr HOST_DEVICE_INLINE bool lt(const Field& xs, const Field& ys)
  {
    ff_storage dummy = {};
    uint32_t carry = sub_limbs<true>(xs.limbs_storage, ys.limbs_storage, dummy);
    return carry;
  }

  static constexpr HOST_DEVICE_INLINE bool is_odd(const Field& xs) { return xs.limbs_storage.limbs[0] & 1; }

  static constexpr HOST_DEVICE_INLINE bool is_even(const Field& xs) { return ~xs.limbs_storage.limbs[0] & 1; }

  // inverse assumes that xs is nonzero
  static constexpr HOST_DEVICE_INLINE Field inverse(const Field& xs)
  {
    constexpr Field one = Field{CONFIG::one};
    constexpr ff_storage modulus = CONFIG::modulus;
    Field u = xs;
    Field v = Field{modulus};
    Field b = one;
    Field c = {};
    while (!(u == one) && !(v == one)) {
      while (is_even(u)) {
        u = div2(u);
        if (is_odd(b)) add_limbs<false>(b.limbs_storage, modulus, b.limbs_storage);
        b = div2(b);
      }
      while (is_even(v)) {
        v = div2(v);
        if (is_odd(c)) add_limbs<false>(c.limbs_storage, modulus, c.limbs_storage);
        c = div2(c);
      }
      if (lt(v, u)) {
        u = u - v;
        b = b - c;
      } else {
        v = v - u;
        c = c - b;
      }
    }
    return (u == one) ? b : c;
  }
};
