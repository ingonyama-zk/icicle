#pragma once

#include "../utils/storage.cuh"
#include "../utils/ptx.cuh"
#include "../utils/host_math.cuh"
#include <random>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

#define HOST_INLINE __host__ __forceinline__
#define DEVICE_INLINE __device__ __forceinline__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__

template <class CONFIG> class Field {
  public:
    static constexpr unsigned TLC = CONFIG::limbs_count;
    static constexpr unsigned NBITS = CONFIG::modulus_bits_count;

    static constexpr HOST_DEVICE_INLINE Field zero() {
      return Field { CONFIG::zero };
    }

    static constexpr HOST_DEVICE_INLINE Field one() {
      return Field { CONFIG::one };
    }

    static constexpr HOST_DEVICE_INLINE Field from(uint32_t value) {
      storage<TLC> scalar;
      scalar.limbs[0] = value;
      for (int i = 1; i < TLC; i++) {
        scalar.limbs[i] = 0;
      }
      return Field { scalar };
    }

    static constexpr HOST_DEVICE_INLINE Field generator_x() {
      return Field { CONFIG::generator_x };
    }

    static constexpr HOST_DEVICE_INLINE Field generator_y() {
      return Field { CONFIG::generator_y };
    }

    static constexpr HOST_INLINE Field omega(uint32_t log_size) {
      // Quick fix to linking issue, permanent fix will follow
      switch (log_size) {
        case 0:
          return Field { CONFIG::one };
        case 1:
          return Field { CONFIG::omega1 };
        case 2:
          return Field { CONFIG::omega2 };
        case 3:
          return Field { CONFIG::omega3 };
        case 4:
          return Field { CONFIG::omega4 };
        case 5:
          return Field { CONFIG::omega5 };
        case 6:
          return Field { CONFIG::omega6 };
        case 7:
          return Field { CONFIG::omega7 };
        case 8:
          return Field { CONFIG::omega8 };
        case 9:
          return Field { CONFIG::omega9 };
        case 10:
          return Field { CONFIG::omega10 };
        case 11:
          return Field { CONFIG::omega11 };
        case 12:
          return Field { CONFIG::omega12 };
        case 13:
          return Field { CONFIG::omega13 };
        case 14:
          return Field { CONFIG::omega14 };
        case 15:
          return Field { CONFIG::omega15 };
        case 16:
          return Field { CONFIG::omega16 };
        case 17:
          return Field { CONFIG::omega17 };
        case 18:
          return Field { CONFIG::omega18 };
        case 19:
          return Field { CONFIG::omega19 };
        case 20:
          return Field { CONFIG::omega20 };
        case 21:
          return Field { CONFIG::omega21 };
        case 22:
          return Field { CONFIG::omega22 };
        case 23:
          return Field { CONFIG::omega23 };
        case 24:
          return Field { CONFIG::omega24 };
        case 25:
          return Field { CONFIG::omega25 };
        case 26:
          return Field { CONFIG::omega26 };
        case 27:
          return Field { CONFIG::omega27 };
        case 28:
          return Field { CONFIG::omega28 };
        case 29:
          return Field { CONFIG::omega29 };
        case 30:
          return Field { CONFIG::omega30 };
        case 31:
          return Field { CONFIG::omega31 };
        case 32:
          return Field { CONFIG::omega32 };        
      }
      return Field { CONFIG::one };
      // return Field { CONFIG::omega[log_size-1] };
    }

    static constexpr HOST_INLINE Field omega_inv(uint32_t log_size) {
      // Quick fix to linking issue, permanent fix will follow
      switch (log_size) {
        case 0:
          return Field { CONFIG::one };
        case 1:
          return Field { CONFIG::omega_inv1 };
        case 2:
          return Field { CONFIG::omega_inv2 };
        case 3:
          return Field { CONFIG::omega_inv3 };
        case 4:
          return Field { CONFIG::omega_inv4 };
        case 5:
          return Field { CONFIG::omega_inv5 };
        case 6:
          return Field { CONFIG::omega_inv6 };
        case 7:
          return Field { CONFIG::omega_inv7 };
        case 8:
          return Field { CONFIG::omega_inv8 };
        case 9:
          return Field { CONFIG::omega_inv9 };
        case 10:
          return Field { CONFIG::omega_inv10 };
        case 11:
          return Field { CONFIG::omega_inv11 };
        case 12:
          return Field { CONFIG::omega_inv12 };
        case 13:
          return Field { CONFIG::omega_inv13 };
        case 14:
          return Field { CONFIG::omega_inv14 };
        case 15:
          return Field { CONFIG::omega_inv15 };
        case 16:
          return Field { CONFIG::omega_inv16 };
        case 17:
          return Field { CONFIG::omega_inv17 };
        case 18:
          return Field { CONFIG::omega_inv18 };
        case 19:
          return Field { CONFIG::omega_inv19 };
        case 20:
          return Field { CONFIG::omega_inv20 };
        case 21:
          return Field { CONFIG::omega_inv21 };
        case 22:
          return Field { CONFIG::omega_inv22 };
        case 23:
          return Field { CONFIG::omega_inv23 };
        case 24:
          return Field { CONFIG::omega_inv24 };
        case 25:
          return Field { CONFIG::omega_inv25 };
        case 26:
          return Field { CONFIG::omega_inv26 };
        case 27:
          return Field { CONFIG::omega_inv27 };
        case 28:
          return Field { CONFIG::omega_inv28 };
        case 29:
          return Field { CONFIG::omega_inv29 };
        case 30:
          return Field { CONFIG::omega_inv30 };
        case 31:
          return Field { CONFIG::omega_inv31 };
        case 32:
          return Field { CONFIG::omega_inv32 };        
      }
      return Field { CONFIG::one };
      // return Field { CONFIG::omega_inv[log_size-1] };
    }

    static constexpr HOST_INLINE Field inv_log_size(uint32_t log_size) {
      // Quick fix to linking issue, permanent fix will follow
      switch (log_size) {
        case 1:
          return Field { CONFIG::inv1 };
        case 2:
          return Field { CONFIG::inv2 };
        case 3:
          return Field { CONFIG::inv3 };
        case 4:
          return Field { CONFIG::inv4 };
        case 5:
          return Field { CONFIG::inv5 };
        case 6:
          return Field { CONFIG::inv6 };
        case 7:
          return Field { CONFIG::inv7 };
        case 8:
          return Field { CONFIG::inv8 };
        case 9:
          return Field { CONFIG::inv9 };
        case 10:
          return Field { CONFIG::inv10 };
        case 11:
          return Field { CONFIG::inv11 };
        case 12:
          return Field { CONFIG::inv12 };
        case 13:
          return Field { CONFIG::inv13 };
        case 14:
          return Field { CONFIG::inv14 };
        case 15:
          return Field { CONFIG::inv15 };
        case 16:
          return Field { CONFIG::inv16 };
        case 17:
          return Field { CONFIG::inv17 };
        case 18:
          return Field { CONFIG::inv18 };
        case 19:
          return Field { CONFIG::inv19 };
        case 20:
          return Field { CONFIG::inv20 };
        case 21:
          return Field { CONFIG::inv21 };
        case 22:
          return Field { CONFIG::inv22 };
        case 23:
          return Field { CONFIG::inv23 };
        case 24:
          return Field { CONFIG::inv24 };
        case 25:
          return Field { CONFIG::inv25 };
        case 26:
          return Field { CONFIG::inv26 };
        case 27:
          return Field { CONFIG::inv27 };
        case 28:
          return Field { CONFIG::inv28 };
        case 29:
          return Field { CONFIG::inv29 };
        case 30:
          return Field { CONFIG::inv30 };
        case 31:
          return Field { CONFIG::inv31 };
        case 32:
          return Field { CONFIG::inv32 };        
      }
      return Field { CONFIG::one };
      // return Field { CONFIG::inv[log_size-1] };
    }

    static constexpr HOST_DEVICE_INLINE Field modulus() {
      return Field { CONFIG::modulus };
    }

  // private:
    typedef storage<TLC> ff_storage;
    typedef storage<2*TLC> ff_wide_storage;

    static constexpr unsigned slack_bits = 32 * TLC - NBITS;

    struct Wide {
      ff_wide_storage limbs_storage;
      
      Field HOST_DEVICE_INLINE get_lower() {
        Field out{};
      #ifdef __CUDA_ARCH__
      #pragma unroll
      #endif
        for (unsigned i = 0; i < TLC; i++)
          out.limbs_storage.limbs[i] = limbs_storage.limbs[i];
        return out;
      }

      Field HOST_DEVICE_INLINE get_higher_with_slack() {
        Field out{};
      #ifdef __CUDA_ARCH__
      #pragma unroll
      #endif
        for (unsigned i = 0; i < TLC; i++) {
        #ifdef __CUDA_ARCH__
          out.limbs_storage.limbs[i] = __funnelshift_lc(limbs_storage.limbs[i + TLC - 1], limbs_storage.limbs[i + TLC], slack_bits);
        #else
          out.limbs_storage.limbs[i] = (limbs_storage.limbs[i + TLC] << slack_bits) + (limbs_storage.limbs[i + TLC - 1] >> (32 - slack_bits));
        #endif
        }
        return out;
      }
    };

    friend HOST_DEVICE_INLINE Wide operator+(Wide xs, const Wide& ys) {   
      Wide rs = {};
      add_limbs<false>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
      return rs;
    }

    // an incomplete impl that assumes that xs > ys
    friend HOST_DEVICE_INLINE Wide operator-(Wide xs, const Wide& ys) {   
      Wide rs = {};
      sub_limbs<false>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
      return rs;
    }

    // return modulus
    template <unsigned MULTIPLIER = 1> static constexpr HOST_DEVICE_INLINE ff_storage get_modulus() {
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

    template <unsigned MULTIPLIER = 1> static constexpr HOST_DEVICE_INLINE ff_wide_storage modulus_wide() {
      return CONFIG::modulus_wide;
    }

    // return m
    static constexpr HOST_DEVICE_INLINE ff_storage get_m() {
      return CONFIG::m;
    }

    // return modulus^2, helpful for ab +/- cd
    template <unsigned MULTIPLIER = 1> static constexpr HOST_DEVICE_INLINE ff_wide_storage get_modulus_squared() {
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

    // add or subtract limbs
    template <bool SUBTRACT, bool CARRY_OUT> 
    static constexpr DEVICE_INLINE uint32_t add_sub_limbs_device(const ff_storage &xs, const ff_storage &ys, ff_storage &rs) {
      const uint32_t *x = xs.limbs;
      const uint32_t *y = ys.limbs;
      uint32_t *r = rs.limbs;
      r[0] = SUBTRACT ? ptx::sub_cc(x[0], y[0]) : ptx::add_cc(x[0], y[0]);
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
      for (unsigned i = 1; i < (CARRY_OUT ? TLC : TLC - 1); i++)
        r[i] = SUBTRACT ? ptx::subc_cc(x[i], y[i]) : ptx::addc_cc(x[i], y[i]);
      if (!CARRY_OUT) {
        r[TLC - 1] = SUBTRACT ? ptx::subc(x[TLC - 1], y[TLC - 1]) : ptx::addc(x[TLC - 1], y[TLC - 1]);
        return 0;
      }
      return SUBTRACT ? ptx::subc(0, 0) : ptx::addc(0, 0);
    }

    template <bool SUBTRACT, bool CARRY_OUT> 
    static constexpr DEVICE_INLINE uint32_t add_sub_limbs_device(const ff_wide_storage &xs, const ff_wide_storage &ys, ff_wide_storage &rs) {
      const uint32_t *x = xs.limbs;
      const uint32_t *y = ys.limbs;
      uint32_t *r = rs.limbs;
      r[0] = SUBTRACT ? ptx::sub_cc(x[0], y[0]) : ptx::add_cc(x[0], y[0]);
    #ifdef __CUDA_ARCH__
    #pragma unroll
    #endif
      for (unsigned i = 1; i < (CARRY_OUT ? 2 * TLC : 2 * TLC - 1); i++)
        r[i] = SUBTRACT ? ptx::subc_cc(x[i], y[i]) : ptx::addc_cc(x[i], y[i]);
      if (!CARRY_OUT) {
        r[2 * TLC - 1] = SUBTRACT ? ptx::subc(x[2 * TLC - 1], y[2 * TLC - 1]) : ptx::addc(x[2 * TLC - 1], y[2 * TLC - 1]);
        return 0;
      }
      return SUBTRACT ? ptx::subc(0, 0) : ptx::addc(0, 0);
    }

    template <bool SUBTRACT, bool CARRY_OUT>
    static constexpr HOST_INLINE uint32_t add_sub_limbs_host(const ff_storage &xs, const ff_storage &ys, ff_storage &rs) {
      const uint32_t *x = xs.limbs;
      const uint32_t *y = ys.limbs;
      uint32_t *r = rs.limbs;
      uint32_t carry = 0;
      host_math::carry_chain<TLC, false, CARRY_OUT> chain;
      for (unsigned i = 0; i < TLC; i++)
        r[i] = SUBTRACT ? chain.sub(x[i], y[i], carry) : chain.add(x[i], y[i], carry);
      return CARRY_OUT ? carry : 0;
    }

    template <bool SUBTRACT, bool CARRY_OUT>
    static constexpr HOST_INLINE uint32_t add_sub_limbs_host(const ff_wide_storage &xs, const ff_wide_storage &ys, ff_wide_storage &rs) {
      const uint32_t *x = xs.limbs;
      const uint32_t *y = ys.limbs;
      uint32_t *r = rs.limbs;
      uint32_t carry = 0;
      host_math::carry_chain<2 * TLC, false, CARRY_OUT> chain;
      for (unsigned i = 0; i < 2 * TLC; i++)
        r[i] = SUBTRACT ? chain.sub(x[i], y[i], carry) : chain.add(x[i], y[i], carry);
      return CARRY_OUT ? carry : 0;
    }

    template <bool CARRY_OUT, typename T> static constexpr HOST_DEVICE_INLINE uint32_t add_limbs(const T &xs, const T &ys, T &rs) {
    #ifdef __CUDA_ARCH__
      return add_sub_limbs_device<false, CARRY_OUT>(xs, ys, rs);
    #else
      return add_sub_limbs_host<false, CARRY_OUT>(xs, ys, rs);
    #endif
    }

    template <bool CARRY_OUT, typename T> static constexpr HOST_DEVICE_INLINE uint32_t sub_limbs(const T &xs, const T &ys, T &rs) {
    #ifdef __CUDA_ARCH__
      return add_sub_limbs_device<true, CARRY_OUT>(xs, ys, rs);
    #else
      return add_sub_limbs_host<true, CARRY_OUT>(xs, ys, rs);
    #endif
    }

    static DEVICE_INLINE void mul_n(uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = TLC) {
    #pragma unroll
      for (size_t i = 0; i < n; i += 2) {
        acc[i] = ptx::mul_lo(a[i], bi);
        acc[i + 1] = ptx::mul_hi(a[i], bi);
      }
    }

    static DEVICE_INLINE void cmad_n(uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = TLC) {
      acc[0] = ptx::mad_lo_cc(a[0], bi, acc[0]);
      acc[1] = ptx::madc_hi_cc(a[0], bi, acc[1]);
    #pragma unroll
      for (size_t i = 2; i < n; i += 2) {
        acc[i] = ptx::madc_lo_cc(a[i], bi, acc[i]);
        acc[i + 1] = ptx::madc_hi_cc(a[i], bi, acc[i + 1]);
      }
    }

    static DEVICE_INLINE void mad_row(uint32_t *odd, uint32_t *even, const uint32_t *a, uint32_t bi, size_t n = TLC) {
      cmad_n(odd, a + 1, bi, n - 2);
      odd[n - 2] = ptx::madc_lo_cc(a[n - 1], bi, 0);
      odd[n - 1] = ptx::madc_hi(a[n - 1], bi, 0);
      cmad_n(even, a, bi, n);
      odd[n - 1] = ptx::addc(odd[n - 1], 0);
    }

    static DEVICE_INLINE void multiply_raw_device(const ff_storage &as, const ff_storage &bs, ff_wide_storage &rs) {
      const uint32_t *a = as.limbs;
      const uint32_t *b = bs.limbs;
      uint32_t *even = rs.limbs;
      __align__(8) uint32_t odd[2 * TLC - 2];
      mul_n(even, a, b[0]);
      mul_n(odd, a + 1, b[0]);
      mad_row(&even[2], &odd[0], a, b[1]);
      size_t i;
    #pragma unroll
      for (i = 2; i < TLC - 1; i += 2) {
        mad_row(&odd[i], &even[i], a, b[i]);
        mad_row(&even[i + 2], &odd[i], a, b[i + 1]);
      }
      // merge |even| and |odd|
      even[1] = ptx::add_cc(even[1], odd[0]);
      for (i = 1; i < 2 * TLC - 2; i++)
        even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
      even[i + 1] = ptx::addc(even[i + 1], 0);
    }

    static HOST_INLINE void multiply_raw_host(const ff_storage &as, const ff_storage &bs, ff_wide_storage &rs) {
      const uint32_t *a = as.limbs;
      const uint32_t *b = bs.limbs;
      uint32_t *r = rs.limbs;
      for (unsigned i = 0; i < TLC; i++) {
        uint32_t carry = 0;
        for (unsigned j = 0; j < TLC; j++)
          r[j + i] = host_math::madc_cc(a[j], b[i], r[j + i], carry);
        r[TLC + i] = carry;
      }
    }

    static HOST_DEVICE_INLINE void multiply_raw(const ff_storage &as, const ff_storage &bs, ff_wide_storage &rs) {
    #ifdef __CUDA_ARCH__
      return multiply_raw_device(as, bs, rs);
    #else
      return multiply_raw_host(as, bs, rs);
    #endif
    }

  public:
    ff_storage limbs_storage;

    HOST_DEVICE_INLINE uint32_t* export_limbs() {
       return (uint32_t *)limbs_storage.limbs;
    }

    HOST_DEVICE_INLINE unsigned get_scalar_digit(unsigned digit_num, unsigned digit_width) {
      const uint32_t limb_lsb_idx = (digit_num*digit_width) / 32;
      const uint32_t shift_bits = (digit_num*digit_width) % 32;
      unsigned rv = limbs_storage.limbs[limb_lsb_idx] >> shift_bits;
      if ((shift_bits + digit_width > 32) && (limb_lsb_idx+1 < TLC)) {
        rv += limbs_storage.limbs[limb_lsb_idx + 1] << (32 - shift_bits);
      }
      rv &= ((1 << digit_width) - 1);
      return rv;
    }

    static HOST_INLINE Field rand_host() {
      std::random_device rd;
      std::mt19937_64 generator(rd());
      std::uniform_int_distribution<unsigned> distribution;
      Field value{};
      for (unsigned i = 0; i < TLC; i++)
        value.limbs_storage.limbs[i] = distribution(generator);
      while (lt(modulus(), value))
        value = value - modulus();
      return value;
    }

    template <unsigned REDUCTION_SIZE = 1> static constexpr HOST_DEVICE_INLINE Field reduce(const Field &xs) {
      if (REDUCTION_SIZE == 0)
        return xs;
      const ff_storage modulus = get_modulus<REDUCTION_SIZE>();
      Field rs = {};
      return sub_limbs<true>(xs.limbs_storage, modulus, rs.limbs_storage) ? xs : rs;
    }

    friend std::ostream& operator<<(std::ostream& os, const Field& xs) {
      std::stringstream hex_string;
      hex_string << std::hex << std::setfill('0');

      for (int i = 0; i < TLC; i++) {
          hex_string << std::setw(8) << xs.limbs_storage.limbs[i];
      }

      os << "0x" << hex_string.str();
      return os;
    }

    friend HOST_DEVICE_INLINE Field operator+(Field xs, const Field& ys) {
      Field rs = {};
      add_limbs<false>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
      return reduce<1>(rs);
    }

    friend HOST_DEVICE_INLINE Field operator-(Field xs, const Field& ys) {
      Field rs = {};
      uint32_t carry = sub_limbs<true>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
      if (carry == 0)
        return rs;
      const ff_storage modulus = get_modulus<1>();
      add_limbs<false>(rs.limbs_storage, modulus, rs.limbs_storage);
      return rs;
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE Wide mul_wide(const Field& xs, const Field& ys) {
      Wide rs = {};
      multiply_raw(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
      return rs;
    }

    friend HOST_DEVICE_INLINE Field operator*(const Field& xs, const Field& ys) {
      Wide xy = mul_wide(xs, ys);
      Field xy_hi = xy.get_higher_with_slack();
      Wide l = {};
      multiply_raw(xy_hi.limbs_storage, get_m(), l.limbs_storage);
      Field l_hi = l.get_higher_with_slack();
      Wide lp = {};
      multiply_raw(l_hi.limbs_storage, get_modulus(), lp.limbs_storage);
      Wide r_wide = xy - lp;
      Wide r_wide_reduced = {};
      uint32_t reduced = sub_limbs<true>(r_wide.limbs_storage, modulus_wide(), r_wide_reduced.limbs_storage);
      r_wide = reduced ? r_wide : r_wide_reduced;
      Field r = r_wide.get_lower();
      return reduce<1>(r);
    }

    friend HOST_DEVICE_INLINE bool operator==(const Field& xs, const Field& ys) {
    #ifdef __CUDA_ARCH__
      const uint32_t *x = xs.limbs_storage.limbs;
      const uint32_t *y = ys.limbs_storage.limbs;
      uint32_t limbs_or = x[0] ^ y[0];
  #pragma unroll
      for (unsigned i = 1; i < TLC; i++)
        limbs_or |= x[i] ^ y[i];
      return limbs_or == 0;
    #else
      for (unsigned i = 0; i < TLC; i++)
      if (xs.limbs_storage.limbs[i] != ys.limbs_storage.limbs[i])
        return false;
      return true;
    #endif
    }

    friend HOST_DEVICE_INLINE bool operator!=(const Field& xs, const Field& ys) {
      return !(xs == ys);
    }

    template <const Field& multiplier, class T> static constexpr HOST_DEVICE_INLINE T mul_const(const T &xs) {
      return mul_unsigned<multiplier.limbs_storage.limbs[0], T>(xs);
    }

    template <uint32_t mutliplier, class T, unsigned REDUCTION_SIZE = 1>
    static constexpr HOST_DEVICE_INLINE T mul_unsigned(const T &xs) {
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
        if (mutliplier & ((1 << (31 - i) - 1) << (i + 1)))
          break;
        temp = temp + temp;
      }
      return rs;
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE Wide sqr_wide(const Field& xs) {
      // TODO: change to a more efficient squaring
      return mul_wide<MODULUS_MULTIPLE>(xs, xs);
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE Field sqr(const Field& xs) {
      // TODO: change to a more efficient squaring
      return xs * xs;
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE Field pow(const Field& xs, uint32_t exp) {
      Field result = Field::one();
      Field base = xs * result; 
      while (exp != 0)
      {
          if ((exp & 1) == 1)
              result = result * base;
          exp >>= 1;
          base = base * base;
      }
      return result;
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE Field neg(const Field& xs) {
      const ff_storage modulus = get_modulus<MODULUS_MULTIPLE>();
      Field rs = {};
      sub_limbs<false>(modulus, xs.limbs_storage, rs.limbs_storage);
      return rs;
    }

    template <unsigned MODULUS_MULTIPLE = 1> 
    static constexpr HOST_DEVICE_INLINE Field div2(const Field &xs) {
      const uint32_t *x = xs.limbs_storage.limbs;
      Field rs = {};
      uint32_t *r = rs.limbs_storage.limbs;
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
      return reduce<MODULUS_MULTIPLE>(rs);
    }

    static constexpr HOST_DEVICE_INLINE bool lt(const Field &xs, const Field &ys) {
      ff_storage dummy = {};
      uint32_t carry = sub_limbs<true>(xs.limbs_storage, ys.limbs_storage, dummy);
      return carry;
    }

    static constexpr HOST_DEVICE_INLINE bool is_odd(const Field &xs) { 
      return xs.limbs_storage.limbs[0] & 1;
    }

    static constexpr HOST_DEVICE_INLINE bool is_even(const Field &xs) { 
      return ~xs.limbs_storage.limbs[0] & 1;
    }

    // inverse assumes that xs is nonzero
    static constexpr HOST_DEVICE_INLINE Field inverse(const Field& xs) {
      constexpr Field one = Field { CONFIG::one };
      constexpr ff_storage modulus = CONFIG::modulus;
      Field u = xs;
      Field v = Field { modulus };
      Field b = one;
      Field c = {};
      while (!(u == one) && !(v == one)) {
        while (is_even(u)) {
          u = div2(u);
          if (is_odd(b))
            add_limbs<false>(b.limbs_storage, modulus, b.limbs_storage);
          b = div2(b);
        }
        while (is_even(v)) {
          v = div2(v);
          if (is_odd(c))
            add_limbs<false>(c.limbs_storage, modulus, c.limbs_storage);
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
      return (u == one) ?  b : c;
    }
};
