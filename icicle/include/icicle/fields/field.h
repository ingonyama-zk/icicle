#pragma once

/**
 * This file contains methods for working with elements of a prime field. It is based on and evolved from Matter Labs'
 * [Zprize
 * submission](https://github.com/matter-labs/z-prize-msm-gpu/blob/main/bellman-cuda-rust/bellman-cuda-sys/native/ff_dispatch_st.h).
 *
 * TODO: DmytroTym: current version needs refactoring (e.g. there's no reason to have different classes Field and
 * ff_storage among other issues). But because this is an internal file and correctness and performance are unaffected,
 * refactoring it is low in the priority list.
 *
 * Documentation of methods is intended to explain inner workings to developers working on icicle. In its current state
 * it mostly explains modular multiplication and related methods. One important quirk of modern CUDA that's affecting
 * most methods is explained by [Niall Emmart](https://youtu.be/KAWlySN7Hm8?si=h7nzDujnvubWXeDX&t=4039). In short, when
 * 64-bit MAD (`r = a * b + c`) instructions get compiled down to SASS (CUDA assembly) they require two-register values
 * `r` and `c` to start from even register (e.g. `r` can live in registers 20 and 21, or 14 and 15, but not 15 and 16).
 * This complicates implementations forcing us to segregate terms into two categories depending on their alignment.
 * Which is where `even` and `odd` arrays across the codebase come from.
 */

#ifdef __CUDACC__
  #include "gpu-utils/sharedmem.h"
  #include "ptx.h"
#endif // __CUDACC__

#include "icicle/errors.h"
#include "host_math.h"
#include "device_math.h"
#include "storage.h"

#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

using namespace icicle;

template <class CONFIG>
class Field
{
public:
  static constexpr unsigned TLC = CONFIG::limbs_count;
  static constexpr unsigned NBITS = CONFIG::modulus_bit_count;

  static constexpr HOST_DEVICE_INLINE Field zero() { return Field{CONFIG::zero}; }

#ifdef BARRET
  static constexpr HOST_DEVICE_INLINE Field one() { return Field{CONFIG::one}; }
#else
  static constexpr HOST_DEVICE_INLINE Field one() { return Field{CONFIG::montgomery_r}; }
#endif

  static constexpr HOST_DEVICE_INLINE Field from(uint32_t value)
  {
    storage<TLC> scalar{};
    scalar.limbs[0] = value;
    for (int i = 1; i < TLC; i++) {
      scalar.limbs[i] = 0;
    }
#ifdef BARRET
    return Field{scalar};
#else
    return to_montgomery(Field{scalar});
#endif
  }

  static HOST_INLINE Field omega(uint32_t logn)
  {
    if (logn == 0) { return one(); }

    if (logn > CONFIG::omegas_count) { THROW_ICICLE_ERR(eIcicleError::INVALID_ARGUMENT, "Field: Invalid omega index"); }
#ifdef BARRET
    Field omega = Field{CONFIG::rou};
#else
    Field omega = to_montgomery(Field{CONFIG::rou});
#endif
    for (int i = 0; i < CONFIG::omegas_count - logn; i++)
      omega = sqr(omega);
    return omega;
  }

  static HOST_INLINE Field omega_inv(uint32_t logn)
  {
    if (logn == 0) { return one(); }

    if (logn > CONFIG::omegas_count) {
      THROW_ICICLE_ERR(eIcicleError::INVALID_ARGUMENT, "Field: Invalid omega_inv index");
    }
#ifdef BARRET
    Field omega = inverse(Field{CONFIG::rou});
#else
    Field omega = inverse(to_montgomery(Field{CONFIG::rou}));
#endif
    for (int i = 0; i < CONFIG::omegas_count - logn; i++)
      omega = sqr(omega);
    return omega;
  }

  static HOST_DEVICE_INLINE Field inv_log_size(uint32_t logn)
  {
    if (logn == 0) { return one(); }
#ifndef __CUDA_ARCH__
    if (logn > CONFIG::omegas_count) THROW_ICICLE_ERR(eIcicleError::INVALID_ARGUMENT, "Field: Invalid inv index");
#else
    if (logn > CONFIG::omegas_count) {
      printf(
        "CUDA ERROR: field.h: error on inv_log_size(logn): logn(=%u) > omegas_count (=%u)", logn, CONFIG::omegas_count);
      assert(false);
    }
#endif // __CUDA_ARCH__
    storage_array<CONFIG::omegas_count, TLC> const inv = CONFIG::inv;
#ifdef BARRET
    return Field{inv.storages[logn - 1]};
#else
    return to_montgomery(Field{inv.storages[logn - 1]});
#endif
  }

  static constexpr HOST_INLINE unsigned get_omegas_count()
  {
    if constexpr (has_member_omegas_count<CONFIG>()) {
      return CONFIG::omegas_count;
    } else {
      return 0;
    }
  }

  template <typename T>
  static constexpr bool has_member_omegas_count()
  {
    return sizeof(T::omegas_count) > 0;
  }

  // private:
  typedef storage<TLC> ff_storage;
  typedef storage<2 * TLC> ff_wide_storage;

  /**
   * A new addition to the config file - \f$ 2^{32 \cdot num\_limbs} - p \f$.
   */
  static constexpr HOST_DEVICE_INLINE ff_storage get_neg_modulus() { return CONFIG::neg_modulus; }

  static constexpr HOST_DEVICE_INLINE ff_storage get_mont_inv_modulus() { return CONFIG::mont_inv_modulus; }
  static constexpr HOST_DEVICE_INLINE ff_storage get_mont_r() { return CONFIG::montgomery_r; }
  static constexpr HOST_DEVICE_INLINE ff_storage get_mont_r_sqr() { return CONFIG::montgomery_r_sqr; }
  static constexpr HOST_DEVICE_INLINE ff_storage get_mont_r_inv() { return CONFIG::montgomery_r_inv; }
  /**
   * A new addition to the config file - the number of times to reduce in [reduce](@ref reduce) function.
   */
  static constexpr HOST_DEVICE_INLINE unsigned num_of_reductions() { return CONFIG::num_of_reductions; }

  static constexpr unsigned slack_bits = 32 * TLC - NBITS;

  struct Wide {
    ff_wide_storage limbs_storage;

    static constexpr Wide HOST_DEVICE_INLINE from_field(const Field& xs)
    {
      Wide out{};
#ifdef __CUDA_ARCH__
      UNROLL
#endif
      for (unsigned i = 0; i < TLC; i++)
        out.limbs_storage.limbs[i] = xs.limbs_storage.limbs[i];
      return out;
    }

    static constexpr Field HOST_DEVICE_INLINE get_lower(const Wide& xs)
    {
      Field out{};
#ifdef __CUDA_ARCH__
      UNROLL
#endif
      for (unsigned i = 0; i < TLC; i++)
        out.limbs_storage.limbs[i] = xs.limbs_storage.limbs[i];
      return out;
    }

    static constexpr Field HOST_DEVICE_INLINE get_higher(const Wide& xs)
    {
      Field out{};
#ifdef __CUDA_ARCH__
      UNROLL
#endif
      for (unsigned i = 0; i < TLC; i++)
        out.limbs_storage.limbs[i] = xs.limbs_storage.limbs[i + TLC];
      return out;
    }

    static constexpr Field HOST_DEVICE_INLINE get_higher_with_slack(const Wide& xs)
    {
      Field out{};
#ifdef __CUDA_ARCH__
      UNROLL
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
      return sub_limbs<2 * TLC, true>(xs.limbs_storage, modulus, rs.limbs_storage) ? xs : rs;
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE Wide neg(const Wide& xs)
    {
      const ff_wide_storage modulus = get_modulus_squared<MODULUS_MULTIPLE>();
      Wide rs = {};
      sub_limbs<2 * TLC, false>(modulus, xs.limbs_storage, rs.limbs_storage);
      return rs;
    }

    friend HOST_DEVICE Wide operator+(Wide xs, const Wide& ys)
    {
      Wide rs = {};
      add_limbs<2 * TLC, false>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
      return sub_modulus_squared<1>(rs);
    }

    friend HOST_DEVICE_INLINE Wide operator-(Wide xs, const Wide& ys)
    {
      Wide rs = {};
      uint32_t carry = sub_limbs<2 * TLC, true>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
      if (carry == 0) return rs;
      const ff_wide_storage modulus = get_modulus_squared<1>();
      add_limbs<2 * TLC, false>(rs.limbs_storage, modulus, rs.limbs_storage);
      return rs;
    }
  };

  // return modulus multiplied by 1, 2 or 4
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

  template <unsigned NLIMBS, bool CARRY_OUT>
  static constexpr HOST_DEVICE_INLINE uint32_t
  add_limbs(const storage<NLIMBS>& xs, const storage<NLIMBS>& ys, storage<NLIMBS>& rs)
  {
#ifdef __CUDA_ARCH__
    return device_math::template add_sub_limbs_device<NLIMBS, false, CARRY_OUT>(xs, ys, rs);
#else
    return host_math::template add_sub_limbs<NLIMBS, false, CARRY_OUT>(xs, ys, rs);
#endif
  }

  template <unsigned NLIMBS, bool CARRY_OUT>
  static constexpr HOST_DEVICE_INLINE uint32_t
  sub_limbs(const storage<NLIMBS>& xs, const storage<NLIMBS>& ys, storage<NLIMBS>& rs)
  {
#ifdef __CUDA_ARCH__
    return device_math::template add_sub_limbs_device<NLIMBS, true, CARRY_OUT>(xs, ys, rs);
#else
    return host_math::template add_sub_limbs<NLIMBS, true, CARRY_OUT>(xs, ys, rs);
#endif
  }

  static HOST_DEVICE_INLINE void multiply_raw(const ff_storage& as, const ff_storage& bs, ff_wide_storage& rs)
  {
#ifdef __CUDA_ARCH__
    return device_math::template multiply_raw_device(as, bs, rs);
#else
    return host_math::template multiply_raw<TLC>(as, bs, rs);
#endif
  }

  static HOST_DEVICE_INLINE void
  multiply_and_add_lsb_neg_modulus_raw(const ff_storage& as, ff_storage& cs, ff_storage& rs)
  {
#ifdef __CUDA_ARCH__
    return device_math::template multiply_and_add_lsb_neg_modulus_raw_device(as, get_neg_modulus(), cs, rs);
#else
    Wide r_wide = {};
    host_math::template multiply_raw<TLC>(as, get_neg_modulus(), r_wide.limbs_storage);
    Field r = Wide::get_lower(r_wide);
    add_limbs<TLC, false>(cs, r.limbs_storage, rs);
#endif
  }

  static HOST_DEVICE_INLINE void multiply_msb_raw(const ff_storage& as, const ff_storage& bs, ff_wide_storage& rs)
  {
#ifdef __CUDA_ARCH__
    return device_math::template multiply_msb_raw_device(as, bs, rs);
#else
    return host_math::template multiply_raw<TLC>(as, bs, rs);
#endif
  }

public:
  ff_storage limbs_storage;

  HOST_DEVICE_INLINE uint32_t* export_limbs() { return (uint32_t*)limbs_storage.limbs; }

  HOST_DEVICE_INLINE unsigned get_scalar_digit(unsigned digit_num, unsigned digit_width) const
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
    while (lt(Field{get_modulus()}, value))
      value = value - Field{get_modulus()};
#ifdef BARRET
    return value;
#else
    return to_montgomery(value);
#endif
  }

  static void rand_host_many(Field* out, int size)
  {
    for (int i = 0; i < size; i++)
      out[i] = rand_host();
  }

  template <unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE Field sub_modulus(const Field& xs)
  {
    if (REDUCTION_SIZE == 0) return xs;
    const ff_storage modulus = get_modulus<REDUCTION_SIZE>();
    Field rs = {};
    return sub_limbs<TLC, true>(xs.limbs_storage, modulus, rs.limbs_storage) ? xs : rs;
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

  friend HOST_DEVICE Field operator+(Field xs, const Field& ys)
  {
    Field rs = {};
    add_limbs<TLC, false>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
    return sub_modulus<1>(rs);
  }

  friend HOST_DEVICE Field operator-(Field xs, const Field& ys)
  {
    Field rs = {};
    uint32_t carry = sub_limbs<TLC, true>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
    if (carry == 0) return rs;
    const ff_storage modulus = get_modulus<1>();
    add_limbs<TLC, false>(rs.limbs_storage, modulus, rs.limbs_storage);
    return rs;
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Wide mul_wide(const Field& xs, const Field& ys)
  {
    Wide rs = {};
    multiply_raw(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
    return rs;
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
  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Field reduce(const Wide& xs) //TODO = add reduce_mont_inplace
  {
    // `xs` is left-shifted by `2 * slack_bits` and higher half is written to `xs_hi`
    Field xs_hi = Wide::get_higher_with_slack(xs);
    Wide l = {};
    multiply_msb_raw(xs_hi.limbs_storage, get_m(), l.limbs_storage); // MSB mult by `m`
    Field l_hi = Wide::get_higher(l);
    Field r = {};
    Field xs_lo = Wide::get_lower(xs);
    // Here we need to compute the lsb of `xs - l \cdot p` and to make use of fused multiply-and-add, we rewrite it as
    // `xs + l \cdot (2^{32 \cdot TLC}-p)` which is the same as original (up to higher limbs which we don't care about).
    multiply_and_add_lsb_neg_modulus_raw(l_hi.limbs_storage, xs_lo.limbs_storage, r.limbs_storage);
    ff_storage r_reduced = {};
    uint32_t carry = 0;
    // As mentioned, either 2 or 1 reduction can be performed depending on the field in question.
    if (num_of_reductions() == 2) {
      carry = sub_limbs<TLC, true>(r.limbs_storage, get_modulus<2>(), r_reduced);
      if (carry == 0) r = Field{r_reduced};
    }
    carry = sub_limbs<TLC, true>(r.limbs_storage, get_modulus<1>(), r_reduced);
    if (carry == 0) r = Field{r_reduced};

    return r;
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Field mont_reduce(const Wide& xs, bool get_higher_half = false)
  {
    //  Field xs_lo = Wide::get_lower(xs);
    //  Field xs_hi = Wide::get_higher(xs);
    //  Wide l1 = {};
    //  Wide l2 = {};
    //  host_math::template multiply_raw<TLC>(xs_lo.limbs_storage, get_m(), l1.limbs_storage);
    //  Field l1_lo = Wide::get_lower(l1);
    //  host_math::template multiply_raw<TLC>(l1_lo.limbs_storage, get_modulus<1>(), l2.limbs_storage);
    //  Field l2_hi = Wide::get_higher(l2);
    //  Field r = {};
    //  add_limbs<TLC, false>(l2_hi.limbs_storage, xs_hi.limbs_storage, r.limbs_storage);

    Field r = get_higher_half ? Wide::get_higher(xs) : Wide::get_lower(xs);
    Field p = Field{get_modulus<1>()};
    if (p.limbs_storage.limbs[TLC - 1] > r.limbs_storage.limbs[TLC - 1]) return r;
    ff_storage r_reduced = {};
    uint64_t carry = 0;
    carry = sub_limbs<TLC, true>(r.limbs_storage, get_modulus<1>(), r_reduced);
    if (carry == 0) r = Field{r_reduced};
    return r;
  }

  HOST_DEVICE Field& operator=(Field const& other)
  {
    for (int i = 0; i < TLC; i++) {
      this->limbs_storage.limbs[i] = other.limbs_storage.limbs[i];
    }
    return *this;
  }

  // #if defined(__CUDACC__)
#if 1
  friend HOST_DEVICE Field operator*(const Field& xs, const Field& ys)
  {
  #ifdef __CUDA_ARCH__ // cuda
    #ifdef BARRET
    Wide xy = mul_wide(xs, ys); // full mult
    return reduce(xy);          // reduce mod p
    // return Wide::get_lower(xy);          // reduce mod p
    #else
      return sub_modulus<1>(Field{device_math::template mulmont_device<TLC>(xs.limbs_storage,ys.limbs_storage,get_modulus<1>(),get_mont_inv_modulus())});
    #endif
    #else
    #ifdef BARRET
      Wide xy = mul_wide(xs, ys); // full mult
      return reduce(xy);          // reduce mod p
      // return Wide::get_lower(xy);
    #else
    return mont_mult(xs,ys);
    #endif
    #endif
  #endif
  }

  static constexpr HOST_INLINE Field mont_mult(const Field& xs, const Field& ys)
  {
    Wide r = {};
    host_math::multiply_mont_64<TLC>(
      xs.limbs_storage.limbs64, ys.limbs_storage.limbs64, get_mont_inv_modulus().limbs64, get_modulus<1>().limbs64,
      r.limbs_storage.limbs64);
    return mont_reduce(r);
  }

  /**
   * @brief Perform  SOS reduction on a number in montgomery representation \p t in range [0,p^2-1] (p is the field's
   * modulus) limiting it to the range [0,2p-1].
   * @param t Number to be reduced. Must be in montgomery rep, and in range [0,p^2-1].
   * @return \p t mod p
   */
  static constexpr HOST_INLINE Field sos_mont_reduce(const Wide& t)
  {
    Wide r = {};
    host_math::sos_mont_reduction_64<TLC>(
      t.limbs_storage.limbs64, get_modulus<1>().limbs64, get_mont_inv_modulus().limbs64, r.limbs_storage.limbs64);
    return mont_reduce(r, /* get_higher_half = */ true);
  }

  // #if defined(__GNUC__) && !defined(__NVCC__) && !defined(__clang__)
  //   #pragma GCC optimize("no-strict-aliasing")
  // #endif

  friend HOST_DEVICE_INLINE Field original_multiplier(const Field& xs, const Field& ys)
  {
    Wide xy = mul_wide(xs, ys); // full mult
    return reduce(xy);          // reduce mod p
  }

#ifdef GARBAGE

  // #include <x86intrin.h>

  /* GNARK CODE START*/
  // those two funcs are copied from bits.go implementation (/usr/local/go/src/math/bits/bits.go)
  static HOST_DEVICE_INLINE void Mul64(uint64_t x, uint64_t y, uint64_t& hi, uint64_t& lo)
  {
    // constexpr uint64_t mask32 = 4294967295ULL; // 2^32 - 1
    // uint64_t x0 = x & mask32;
    // uint64_t x1 = x >> 32;
    // uint64_t y0 = y & mask32;
    // uint64_t y1 = y >> 32;
    // uint64_t w0 = x0 * y0;
    // uint64_t t = x1 * y0 + w0 >> 32;
    // uint64_t w1 = t & mask32;
    // uint64_t w2 = t >> 32;
    // w1 += x0 * y1;
    // hi = x1 * y1 + w2 + w1 >> 32;
    // lo = x * y;

    // #if defined(__GNUC__) || defined(__clang__)
    // lo = _umul128(x, y, &hi);
    // #else
    __uint128_t result = static_cast<__uint128_t>(x) * y;
    hi = static_cast<uint64_t>(result >> 64);
    lo = static_cast<uint64_t>(result);
    // #endif
  }

  // #if defined(__GNUC__) || defined(__clang__)
  // #include <x86intrin.h>
  // #endif

  static HOST_DEVICE_INLINE void Add64(uint64_t x, uint64_t y, uint64_t carry, uint64_t& sum, uint64_t& carry_out)
  {
    // #if defined(__GNUC__) || defined(__clang__)
    // carry_out = _addcarry_u64(carry, x, y, &sum);
    // #else
    sum = x + y + carry;
    carry_out = ((x & y) | ((x | y) & ~sum)) >> 63;
    // #endif
  }

  static HOST_DEVICE_INLINE void Sub64(uint64_t x, uint64_t y, uint64_t borrow, uint64_t& diff, uint64_t& borrowOut)
  {
    // #if defined(__GNUC__) || defined(__clang__)
    // borrowOut = _subborrow_u64(borrow, x, y, &diff);
    // #else
    diff = x - y - borrow;
    // See Sub32 for the bit logic.
    borrowOut = ((~x & y) | (~(x ^ y) & diff)) >> 63;
    // #endif
  }

  static HOST_DEVICE_INLINE bool smallerThanModulus(const Field& z)
  {
    // for bn254 specifically
    constexpr uint64_t q0 = 4891460686036598785ULL;
    constexpr uint64_t q1 = 2896914383306846353ULL;
    constexpr uint64_t q2 = 13281191951274694749ULL;
    constexpr uint64_t q3 = 3486998266802970665ULL;
    return (
      z.limbs_storage.limbs64[3] < q3 ||
      (z.limbs_storage.limbs64[3] == q3 &&
       (z.limbs_storage.limbs64[2] < q2 ||
        (z.limbs_storage.limbs64[2] == q2 &&
         (z.limbs_storage.limbs64[1] < q1 ||
          (z.limbs_storage.limbs64[1] == q1 && (z.limbs_storage.limbs64[0] < q0)))))));
  }

  // #define WITH_MONT_CONVERSIONS

  #ifdef WITH_MONT_CONVERSIONS
  friend HOST_DEVICE Field operator*(const Field& x_orig, const Field& y_orig)
  #else
  friend HOST_DEVICE Field operator*(const Field& x, const Field& y)
  #endif
  {
    // for bn254 specifically
    constexpr uint64_t qInvNeg = 14042775128853446655ULL;
    constexpr uint64_t q0 = 4891460686036598785ULL;
    constexpr uint64_t q1 = 2896914383306846353ULL;
    constexpr uint64_t q2 = 13281191951274694749ULL;
    constexpr uint64_t q3 = 3486998266802970665ULL;

  #ifdef WITH_MONT_CONVERSIONS
    // auto x = original_multiplier(x_orig, original_multiplier(Field{CONFIG::montgomery_r},
    // Field{CONFIG::montgomery_r})); auto y = original_multiplier(y_orig,
    // original_multiplier(Field{CONFIG::montgomery_r}, Field{CONFIG::montgomery_r}));
    auto x = original_multiplier(x_orig, Field{CONFIG::montgomery_r});
    auto y = original_multiplier(y_orig, Field{CONFIG::montgomery_r});
  #endif

    Field z{};
    uint64_t t0, t1, t2, t3;
    uint64_t u0, u1, u2, u3;

    {
      uint64_t c0, c1, c2, _;
      uint64_t v = x.limbs_storage.limbs64[0];
      Mul64(v, y.limbs_storage.limbs64[0], u0, t0);
      Mul64(v, y.limbs_storage.limbs64[1], u1, t1);
      Mul64(v, y.limbs_storage.limbs64[2], u2, t2);
      Mul64(v, y.limbs_storage.limbs64[3], u3, t3);
      Add64(u0, t1, 0, t1, c0);
      Add64(u1, t2, c0, t2, c0);
      Add64(u2, t3, c0, t3, c0);
      Add64(u3, 0, c0, c2, _);

      uint64_t m = qInvNeg * t0;

      Mul64(m, q0, u0, c1);
      Add64(t0, c1, 0, _, c0);
      Mul64(m, q1, u1, c1);
      Add64(t1, c1, c0, t0, c0);
      Mul64(m, q2, u2, c1);
      Add64(t2, c1, c0, t1, c0);
      Mul64(m, q3, u3, c1);

      Add64(0, c1, c0, t2, c0);
      Add64(u3, 0, c0, u3, _);
      Add64(u0, t0, 0, t0, c0);
      Add64(u1, t1, c0, t1, c0);
      Add64(u2, t2, c0, t2, c0);
      Add64(c2, 0, c0, c2, _);
      Add64(t3, t2, 0, t2, c0);
      Add64(u3, c2, c0, t3, _);
    }

    {
      uint64_t c0, c1, c2, _;
      uint64_t v = x.limbs_storage.limbs64[1];
      Mul64(v, y.limbs_storage.limbs64[0], u0, c1);
      Add64(c1, t0, 0, t0, c0);
      Mul64(v, y.limbs_storage.limbs64[1], u1, c1);
      Add64(c1, t1, c0, t1, c0);
      Mul64(v, y.limbs_storage.limbs64[2], u2, c1);
      Add64(c1, t2, c0, t2, c0);
      Mul64(v, y.limbs_storage.limbs64[3], u3, c1);
      Add64(c1, t3, c0, t3, c0);

      Add64(0, 0, c0, c2, _);
      Add64(u0, t1, 0, t1, c0);
      Add64(u1, t2, c0, t2, c0);
      Add64(u2, t3, c0, t3, c0);
      Add64(u3, c2, c0, c2, _);

      uint64_t m = qInvNeg * t0;

      Mul64(m, q0, u0, c1);
      Add64(t0, c1, 0, _, c0);
      Mul64(m, q1, u1, c1);
      Add64(t1, c1, c0, t0, c0);
      Mul64(m, q2, u2, c1);
      Add64(t2, c1, c0, t1, c0);
      Mul64(m, q3, u3, c1);

      Add64(0, c1, c0, t2, c0);
      Add64(u3, 0, c0, u3, _);
      Add64(u0, t0, 0, t0, c0);
      Add64(u1, t1, c0, t1, c0);
      Add64(u2, t2, c0, t2, c0);
      Add64(c2, 0, c0, c2, _);
      Add64(t3, t2, 0, t2, c0);
      Add64(u3, c2, c0, t3, _);
    }

    {
      uint64_t c0, c1, c2, _;
      uint64_t v = x.limbs_storage.limbs64[2];
      Mul64(v, y.limbs_storage.limbs64[0], u0, c1);
      Add64(c1, t0, 0, t0, c0);
      Mul64(v, y.limbs_storage.limbs64[1], u1, c1);
      Add64(c1, t1, c0, t1, c0);
      Mul64(v, y.limbs_storage.limbs64[2], u2, c1);
      Add64(c1, t2, c0, t2, c0);
      Mul64(v, y.limbs_storage.limbs64[3], u3, c1);
      Add64(c1, t3, c0, t3, c0);

      Add64(0, 0, c0, c2, _);
      Add64(u0, t1, 0, t1, c0);
      Add64(u1, t2, c0, t2, c0);
      Add64(u2, t3, c0, t3, c0);
      Add64(u3, c2, c0, c2, _);

      uint64_t m = qInvNeg * t0;

      Mul64(m, q0, u0, c1);
      Add64(t0, c1, 0, _, c0);
      Mul64(m, q1, u1, c1);
      Add64(t1, c1, c0, t0, c0);
      Mul64(m, q2, u2, c1);
      Add64(t2, c1, c0, t1, c0);
      Mul64(m, q3, u3, c1);

      Add64(0, c1, c0, t2, c0);
      Add64(u3, 0, c0, u3, _);
      Add64(u0, t0, 0, t0, c0);
      Add64(u1, t1, c0, t1, c0);
      Add64(u2, t2, c0, t2, c0);
      Add64(c2, 0, c0, c2, _);
      Add64(t3, t2, 0, t2, c0);
      Add64(u3, c2, c0, t3, _);
    }

    {
      uint64_t c0, c1, c2, _;
      uint64_t v = x.limbs_storage.limbs64[3];
      Mul64(v, y.limbs_storage.limbs64[0], u0, c1);
      Add64(c1, t0, 0, t0, c0);
      Mul64(v, y.limbs_storage.limbs64[1], u1, c1);
      Add64(c1, t1, c0, t1, c0);
      Mul64(v, y.limbs_storage.limbs64[2], u2, c1);
      Add64(c1, t2, c0, t2, c0);
      Mul64(v, y.limbs_storage.limbs64[3], u3, c1);
      Add64(c1, t3, c0, t3, c0);

      Add64(0, 0, c0, c2, _);
      Add64(u0, t1, 0, t1, c0);
      Add64(u1, t2, c0, t2, c0);
      Add64(u2, t3, c0, t3, c0);
      Add64(u3, c2, c0, c2, _);

      uint64_t m = qInvNeg * t0;

      Mul64(m, q0, u0, c1);
      Add64(t0, c1, 0, _, c0);
      Mul64(m, q1, u1, c1);
      Add64(t1, c1, c0, t0, c0);
      Mul64(m, q2, u2, c1);
      Add64(t2, c1, c0, t1, c0);
      Mul64(m, q3, u3, c1);

      Add64(0, c1, c0, t2, c0);
      Add64(u3, 0, c0, u3, _);
      Add64(u0, t0, 0, t0, c0);
      Add64(u1, t1, c0, t1, c0);
      Add64(u2, t2, c0, t2, c0);
      Add64(c2, 0, c0, c2, _);
      Add64(t3, t2, 0, t2, c0);
      Add64(u3, c2, c0, t3, _);
    }

    z.limbs_storage.limbs64[0] = t0;
    z.limbs_storage.limbs64[1] = t1;
    z.limbs_storage.limbs64[2] = t2;
    z.limbs_storage.limbs64[3] = t3;

    if (smallerThanModulus(z)) {
      uint64_t b, _;
      Sub64(z.limbs_storage.limbs64[0], q0, 0, z.limbs_storage.limbs64[0], b);
      Sub64(z.limbs_storage.limbs64[1], q1, b, z.limbs_storage.limbs64[1], b);
      Sub64(z.limbs_storage.limbs64[2], q2, b, z.limbs_storage.limbs64[2], b);
      Sub64(z.limbs_storage.limbs64[3], q3, b, z.limbs_storage.limbs64[3], _);
    }

  #ifdef WITH_MONT_CONVERSIONS
    z = original_multiplier(z, Field{CONFIG::montgomery_r_inv});
      // z = original_multiplier(z, original_multiplier(Field{CONFIG::montgomery_r_inv},
      // Field{CONFIG::montgomery_r_inv}));
  #endif
    return z;
  }
#endif

  // #if defined(__GNUC__) && !defined(__NVCC__) && !defined(__clang__)
  //   #pragma GCC reset_options
  // #endif

  /*GNARK CODE END*/

  friend HOST_DEVICE bool operator==(const Field& xs, const Field& ys)
  {
#ifdef __CUDA_ARCH__
    const uint32_t* x = xs.limbs_storage.limbs;
    const uint32_t* y = ys.limbs_storage.limbs;
    uint32_t limbs_or = x[0] ^ y[0];
    UNROLL
    for (unsigned i = 1; i < TLC; i++)
      limbs_or |= x[i] ^ y[i];
    return limbs_or == 0;
#else
    for (unsigned i = 0; i < TLC; i++)
      if (xs.limbs_storage.limbs[i] != ys.limbs_storage.limbs[i]) return false;
    return true;
#endif
  }

  friend HOST_DEVICE bool operator!=(const Field& xs, const Field& ys) { return !(xs == ys); }

  template <const Field& multiplier>
  static HOST_DEVICE_INLINE Field mul_const(const Field& xs)
  {
    Field mul = multiplier;
    #ifdef BARRET
    static bool is_u32 = true;
#ifdef __CUDA_ARCH__
    UNROLL
#endif
    for (unsigned i = 1; i < TLC; i++)
      is_u32 &= (mul.limbs_storage.limbs[i] == 0);

    if (is_u32) return mul_unsigned<multiplier.limbs_storage.limbs[0], Field>(xs);
    #endif
    #ifndef BARRET
    mul = to_montgomery(mul);
    #endif
    return mul * xs;
  }

  template <uint32_t multiplier, class T, unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE T mul_unsigned(const T& xs)
  {
    T rs = {};
    T temp = xs;
    bool is_zero = true;
#ifdef __CUDA_ARCH__
    UNROLL
#endif
    for (unsigned i = 0; i < 32; i++) {
      if (multiplier & (1 << i)) {
        rs = is_zero ? temp : (rs + temp);
        is_zero = false;
      }
      if (multiplier & ((1 << (31 - i - 1)) << (i + 1))) break;
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
#ifdef BARRET
  static constexpr HOST_DEVICE_INLINE Field to_montgomery(const Field& xs) { return xs * Field{CONFIG::montgomery_r}; }
  static constexpr HOST_DEVICE_INLINE Field from_montgomery(const Field& xs)
  {
    return xs * Field{CONFIG::montgomery_r_inv};
  }
#else
  static constexpr HOST_DEVICE_INLINE Field to_montgomery(const Field& xs)
  {
    return xs * Field{CONFIG::montgomery_r_sqr};
  }
  static constexpr HOST_DEVICE_INLINE Field from_montgomery(const Field& xs) { return xs * Field{CONFIG::one}; }
#endif

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE Field neg(const Field& xs)
  {
    const ff_storage modulus = get_modulus<MODULUS_MULTIPLE>();
    Field rs = {};
    sub_limbs<TLC, false>(modulus, xs.limbs_storage, rs.limbs_storage);
    return rs;
  }

  // Assumes the number is even!
  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Field div2(const Field& xs)
  {
    const uint32_t* x = xs.limbs_storage.limbs;
    Field rs = {};
    uint32_t* r = rs.limbs_storage.limbs;
    if constexpr (TLC > 1) {
#ifdef __CUDA_ARCH__
      UNROLL
#endif
      for (unsigned i = 0; i < TLC - 1; i++) {
#ifdef __CUDA_ARCH__
        r[i] = __funnelshift_rc(x[i], x[i + 1], 1);
#else
        r[i] = (x[i] >> 1) | (x[i + 1] << 31);
#endif
      }
    }
    r[TLC - 1] = x[TLC - 1] >> 1;
    return sub_modulus<MODULUS_MULTIPLE>(rs);
  }

  static constexpr HOST_DEVICE_INLINE bool lt(const Field& xs, const Field& ys)
  {
    ff_storage dummy = {};
    uint32_t carry = sub_limbs<TLC, true>(xs.limbs_storage, ys.limbs_storage, dummy);
    return carry;
  }

  static constexpr HOST_DEVICE_INLINE bool is_odd(const Field& xs) { return xs.limbs_storage.limbs[0] & 1; }

  static constexpr HOST_DEVICE_INLINE bool is_even(const Field& xs) { return ~xs.limbs_storage.limbs[0] & 1; }

  static constexpr HOST_DEVICE Field inverse(const Field& xs)
  {
    if (xs == zero()) return zero();
    constexpr Field one = {1};
    constexpr ff_storage modulus = CONFIG::modulus;
#ifdef BARRET
    Field u = xs;
#else
    Field u = from_montgomery(xs);
#endif
    Field v = Field{modulus};
    Field b = one;
    Field c = {};
    while (!(u == one) && !(v == one)) {
      while (is_even(u)) {
        u = div2(u);
        if (is_odd(b)) add_limbs<TLC, false>(b.limbs_storage, modulus, b.limbs_storage);
        b = div2(b);
      }
      while (is_even(v)) {
        v = div2(v);
        if (is_odd(c)) add_limbs<TLC, false>(c.limbs_storage, modulus, c.limbs_storage);
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
#ifdef BARRET
    return (u == one) ? b : c;
#else
    return (u == one) ? to_montgomery(b) : to_montgomery(c);
#endif
  }

  static constexpr HOST_DEVICE Field pow(Field base, int exp)
  {
    Field res = one();
    while (exp > 0) {
      if (exp & 1) res = res * base;
      base = base * base;
      exp >>= 1;
    }
    return res;
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