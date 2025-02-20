#pragma once

// CUDA compiles both host and device math. CPU needs only host math.
#ifdef __CUDACC__
  #include "gpu-utils/sharedmem.h"
  #include "cuda_math.h"
#endif // __CUDACC__
#include "icicle/math/host_math.h"

#include "icicle/errors.h"
#include "icicle/utils/rand_gen.h"
#include "icicle/math/storage.h"

#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <cassert>

#ifdef __CUDA_ARCH__
namespace icicle_math = cuda_math;
#else
namespace icicle_math = host_math;
#endif

// NOTE: This class implements modular-arithmetics (via operator overloading) for the derived class type!
//       This is useful as this math is used both for prime-fields and composite-integer-rings
template <typename Derived, typename CONFIG>
class ModArith
{
public:
  static constexpr unsigned TLC = CONFIG::limbs_count;
  static constexpr unsigned NBITS = CONFIG::modulus_bit_count;

  static constexpr HOST_DEVICE_INLINE Derived zero() { return Derived{CONFIG::zero}; }

  static constexpr HOST_DEVICE_INLINE Derived one() { return Derived{CONFIG::one}; }

  static constexpr HOST_DEVICE_INLINE Derived from(uint32_t value)
  {
    storage<TLC> scalar{};
    scalar.limbs[0] = value;
    for (int i = 1; i < TLC; i++) {
      scalar.limbs[i] = 0;
    }
    return Derived{scalar};
  }

  static constexpr HOST_DEVICE_INLINE Derived from_u64(uint64_t value)
  {
    storage<TLC> scalar{};
    scalar.limbs[0] = value;
    scalar.limbs[1] = value >> 32;
    for (int i = 2; i < TLC; i++) {
      scalar.limbs[i] = 0;
    }
    return Derived{scalar};
  }

  static HOST_INLINE Derived omega(uint32_t logn)
  {
    if (logn == 0) { return Derived{CONFIG::one}; }

    if (logn > CONFIG::omegas_count) {
      THROW_ICICLE_ERR(icicle::eIcicleError::INVALID_ARGUMENT, "ModArith: Invalid omega index");
    }

    Derived omega = Derived{CONFIG::rou};
    for (int i = 0; i < CONFIG::omegas_count - logn; i++)
      omega = sqr(omega);
    return omega;
  }

  static HOST_INLINE Derived omega_inv(uint32_t logn)
  {
    if (logn == 0) { return Derived{CONFIG::one}; }

    if (logn > CONFIG::omegas_count) {
      THROW_ICICLE_ERR(icicle::eIcicleError::INVALID_ARGUMENT, "ModArith: Invalid omega_inv index");
    }

    Derived omega = inverse(Derived{CONFIG::rou});
    for (int i = 0; i < CONFIG::omegas_count - logn; i++)
      omega = sqr(omega);
    return omega;
  }

  static HOST_INLINE Derived
  hex_str2scalar(const std::string& in_str) // The input string should be in a hex format - 0xABCD....
  {
    assert(in_str.substr(0, 2) == "0x" && "The input string is not in hex format!");
    std::string tmp_str = in_str.substr(2); // Strip "0x" from the string.
    int length = tmp_str.length();
    // Split string into chunks of 8 chars (for uint32_t) and store in scalar storage.
    storage<TLC> scalar{};
    // for (int str_idx=((int)((length-8)/8))*8, limb_idx = 0; str_idx>=0; str_idx-=8, limb_idx++) {   //
    // ((int)((length-8)/8))*8 is for case if length<8.
    for (int str_idx = length - 8, limb_idx = 0; str_idx >= -7;
         str_idx -= 8, limb_idx++) { // ((int)((length-8)/8))*8 is for case if length<8.
      if (str_idx < 0) {
        scalar.limbs[limb_idx] = strtoul(tmp_str.substr(0, str_idx + 8).c_str(), nullptr, 16);
      } else {
        scalar.limbs[limb_idx] = strtoul(tmp_str.substr(str_idx, 8).c_str(), nullptr, 16);
      }
    }
    return Derived{scalar};
  }

  static HOST_DEVICE_INLINE Derived inv_log_size(uint32_t logn)
  {
    if (logn == 0) { return Derived{CONFIG::one}; }
    icicle_math::index_err(logn, CONFIG::omegas_count); // check if the requested size is within the valid range
    storage_array<CONFIG::omegas_count, TLC> const inv = CONFIG::inv;
    return Derived{inv.storages[logn - 1]};
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

  /**
   * A new addition to the config file - the number of times to reduce in [reduce](@ref reduce) function.
   */
  static constexpr HOST_DEVICE_INLINE unsigned num_of_reductions() { return CONFIG::num_of_reductions; }

  static constexpr unsigned slack_bits = 32 * TLC - NBITS;

  struct Wide {
    ff_wide_storage limbs_storage;

    static constexpr Wide HOST_DEVICE_INLINE from_field(const Derived& xs)
    {
      Wide out{};
#ifdef __CUDA_ARCH__
      UNROLL
#else
  #pragma unroll
#endif
      for (unsigned i = 0; i < TLC; i++)
        out.limbs_storage.limbs[i] = xs.limbs_storage.limbs[i];
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

  // access precomputed values for first step of the from storage function (see below)
  static HOST_DEVICE_INLINE Derived get_reduced_digit_for_storage_reduction(int i)
  {
    storage_array<CONFIG::reduced_digits_count, TLC> const reduced_digits = CONFIG::reduced_digits;
    return Derived{reduced_digits.storages[i]};
  }

  // access precomputed values for second step of the from storage function (see below)
  static HOST_DEVICE_INLINE storage<2 * TLC + 2> get_mod_sub_for_storage_reduction(int i)
  {
    storage_array<CONFIG::mod_subs_count, 2 * TLC + 2> const mod_subs = CONFIG::mod_subs;
    return mod_subs.storages[i];
  }

  template <unsigned NLIMBS, bool CARRY_OUT>
  static constexpr HOST_DEVICE_INLINE uint32_t
  add_limbs(const storage<NLIMBS>& xs, const storage<NLIMBS>& ys, storage<NLIMBS>& rs)
  {
    return icicle_math::template add_sub_limbs<NLIMBS, false, CARRY_OUT>(xs, ys, rs);
  }

  template <unsigned NLIMBS, bool CARRY_OUT>
  static constexpr HOST_DEVICE_INLINE uint32_t
  sub_limbs(const storage<NLIMBS>& xs, const storage<NLIMBS>& ys, storage<NLIMBS>& rs)
  {
    return icicle_math::template add_sub_limbs<NLIMBS, true, CARRY_OUT>(xs, ys, rs);
  }

  static HOST_DEVICE_INLINE void multiply_raw(const ff_storage& as, const ff_storage& bs, ff_wide_storage& rs)
  {
    return icicle_math::template multiply_raw<TLC>(as, bs, rs);
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

  template <unsigned NLIMBS>
  static HOST_INLINE storage<NLIMBS> rand_storage(unsigned non_zero_limbs = NLIMBS)
  {
    std::uniform_int_distribution<unsigned> distribution;
    storage<NLIMBS> value{};
    for (unsigned i = 0; i < non_zero_limbs; i++)
      value.limbs[i] = distribution(rand_generator);
    return value;
  }

  // NOTE this function is used for test and examples - it assumed it is executed on a single-thread (no two threads
  // accessing rand_generator at the same time)
  static HOST_INLINE Derived rand_host()
  {
    std::uniform_int_distribution<unsigned> distribution;
    Derived value{};
    for (unsigned i = 0; i < TLC; i++)
      value.limbs_storage.limbs[i] = distribution(rand_generator);
    while (lt(Derived{get_modulus()}, value))
      value = value - Derived{get_modulus()};
    return value;
  }

  static void rand_host_many(Derived* out, int size)
  {
    for (int i = 0; i < size; i++)
      out[i] = rand_host();
  }

  template <unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE Derived sub_modulus(const Derived& xs)
  {
    if (REDUCTION_SIZE == 0) return xs;
    const ff_storage modulus = get_modulus<REDUCTION_SIZE>();
    Derived rs = {};
    return sub_limbs<TLC, true>(xs.limbs_storage, modulus, rs.limbs_storage) ? xs : rs;
  }

  friend std::ostream& operator<<(std::ostream& os, const Derived& xs)
  {
    std::stringstream hex_string;
    hex_string << std::hex << std::setfill('0');

    for (int i = 0; i < TLC; i++) {
      hex_string << std::setw(8) << xs.limbs_storage.limbs[TLC - i - 1];
    }

    os << "0x" << hex_string.str();
    return os;
  }

  friend HOST_DEVICE Derived operator+(Derived xs, const Derived& ys)
  {
    Derived rs = {};
    add_limbs<TLC, false>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
    return sub_modulus<1>(rs);
  }

  friend HOST_DEVICE Derived operator-(Derived xs, const Derived& ys)
  {
    Derived rs = {};
    uint32_t carry = sub_limbs<TLC, true>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
    if (carry == 0) return rs;
    const ff_storage modulus = get_modulus<1>();
    add_limbs<TLC, false>(rs.limbs_storage, modulus, rs.limbs_storage);
    return rs;
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Wide mul_wide(const Derived& xs, const Derived& ys)
  {
    Wide rs = {};
    multiply_raw(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
    return rs;
  }

  /**
   * This method reduces a Wide number `xs` modulo `p` and returns the result as a ModArith element.
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
  static constexpr HOST_DEVICE_INLINE Derived reduce(const Wide& xs)
  {
    return Derived{icicle_math::template barrett_reduce<TLC, slack_bits, num_of_reductions()>(
      xs.limbs_storage, get_m(), get_modulus(), get_modulus<2>(), get_neg_modulus())};
  }

  /* This function receives a storage object (currently supports up to 576 bits) and reduces it to a field element
  between 0 and p. This is done using 3 steps:
  1. Splitting the number into TLC sized digits - xs = x_i * p_i = x_i * 2^(TLC*32*i).
  p_i are precomputed modulo p and so the first step is performed multiplying by p_i and accumultaing.
  At the end of this step the number is reduced from NLIMBS to 2*TLC+1 (assuming less than 2^32 additions).
  2. The second step subtracts a single precomputed multiple of p in ordr to reduce the number into the range 0<x<2^(2n)
  where n is the modulus bit count. This step makes use of a look-up table that looks at the top bits of the number (it
  is enough to look at the bits from 2^(2n-1) and above).
  3. The final step is the regular barrett reduction that reduces from the range 0<x<2^(2n) down to 0<x<p. */
  template <unsigned NLIMBS>
  static constexpr HOST_DEVICE_INLINE Derived from(const storage<NLIMBS>& xs)
  {
    static_assert(NLIMBS * 32 <= 576); // for now we support up to 576 bits
    storage<2 * TLC + 2> rs = {}; // we use 2*TLC+2 and not 2*TLC+1 because for now we don't support an odd number of
                                  // limbs in the storage struct
    int constexpr size = NLIMBS / TLC;
    // first reduction step:
    for (int i = 0; i < size; i++) // future optimization - because we assume a maximum value for size anyway, this loop
                                   // can be unrolled with potential performance benefits
    {
      const Derived& xi = *reinterpret_cast<const Derived*>(xs.limbs + i * TLC); // use casting instead of copying
      Derived pi = get_reduced_digit_for_storage_reduction(i); // use precomputed values - pi = 2^(TLC*32*i) % p
      storage<2 * TLC + 2> temp = {};
      storage<2 * TLC>& temp_storage = *reinterpret_cast<storage<2 * TLC>*>(temp.limbs);
      icicle_math::template multiply_raw<TLC>(xi.limbs_storage, pi.limbs_storage, temp_storage); // multiplication
      icicle_math::template add_sub_limbs<2 * TLC + 2, false, false>(rs, temp, rs);              // accumulation
    }
    int constexpr extra_limbs = NLIMBS - TLC * size;
    if constexpr (extra_limbs > 0) { // handle the extra limbs (when TLC does not divide NLIMBS)
      const storage<extra_limbs>& xi = *reinterpret_cast<const storage<extra_limbs>*>(xs.limbs + size * TLC);
      Derived pi = get_reduced_digit_for_storage_reduction(size);
      storage<2 * TLC + 2> temp = {};
      storage<extra_limbs + TLC>& temp_storage = *reinterpret_cast<storage<extra_limbs + TLC>*>(temp.limbs);
      icicle_math::template multiply_raw<extra_limbs, TLC>(xi, pi.limbs_storage, temp_storage); // multiplication
      icicle_math::template add_sub_limbs<2 * TLC + 2, false, false>(rs, temp, rs);             // accumulation
    }
    // second reduction step: - an alternative for this step would be to use the barret reduction straight away but with
    // a larger value of m.
    unsigned constexpr msbits_count = 2 * TLC * 32 - (2 * NBITS - 1);
    unsigned top_bits = (rs.limbs[2 * TLC] << msbits_count) + (rs.limbs[2 * TLC - 1] >> (32 - msbits_count));
    icicle_math::template add_sub_limbs<2 * TLC + 2, true, false>(
      rs, get_mod_sub_for_storage_reduction(top_bits),
      rs); // subtracting the precomputed multiple of p from the look-up table
    // third and final step:
    storage<2 * TLC>& res = *reinterpret_cast<storage<2 * TLC>*>(rs.limbs);
    return reduce(Wide{res}); // finally, use barret reduction
  }

  /* This is the non-template version of the from(storage) function above. It receives an array of bytes and its size
  and returns a field element after modular reduction. For now we support up to 576 bits. */
  static constexpr HOST_DEVICE_INLINE Derived from(const std::byte* in, unsigned nof_bytes)
  {
    storage<2 * TLC + 2> rs = {}; // we use 2*TLC+2 and not 2*TLC+1 because for now we don't support an odd number of
                                  // limbs in the storage struct
    unsigned constexpr bytes_per_field = TLC * 4;
    int size = nof_bytes / bytes_per_field;
    // first reduction step:
    for (int i = 0; i < size; i++) {
      const Derived& xi = *reinterpret_cast<const Derived*>(in + i * bytes_per_field); // use casting instead of copying
      Derived pi = get_reduced_digit_for_storage_reduction(i); // use precomputed values - pi = 2^(TLC*32*i) % p
      storage<2 * TLC + 2> temp = {};
      storage<2 * TLC>& temp_storage = *reinterpret_cast<storage<2 * TLC>*>(temp.limbs);
      icicle_math::template multiply_raw<TLC>(xi.limbs_storage, pi.limbs_storage, temp_storage); // multiplication
      icicle_math::template add_sub_limbs<2 * TLC + 2, false, false>(rs, temp, rs);              // accumulation
    }
    int extra_bytes = nof_bytes - bytes_per_field * size;
    if (extra_bytes > 0) { // handle the extra limbs (when TLC does not divide NLIMBS)
      std::byte final_bytes[bytes_per_field] = {};
      for (int j = 0; j < extra_bytes; j++) // this copy cannot be avoided in the non-template version
      {
        final_bytes[j] = in[size * bytes_per_field + j];
      }
      const storage<TLC>& xi = *reinterpret_cast<const storage<TLC>*>(final_bytes);
      Derived pi = get_reduced_digit_for_storage_reduction(size);
      storage<2 * TLC + 2> temp = {};
      storage<2 * TLC>& temp_storage = *reinterpret_cast<storage<2 * TLC>*>(temp.limbs);
      icicle_math::template multiply_raw<TLC>(xi, pi.limbs_storage, temp_storage);  // multiplication
      icicle_math::template add_sub_limbs<2 * TLC + 2, false, false>(rs, temp, rs); // accumulation
    }
    // second reduction step: - an alternative for this step would be to use the barret reduction straight away but with
    // a larger value of m.
    unsigned constexpr msbits_count = 2 * TLC * 32 - (2 * NBITS - 1);
    unsigned top_bits = (rs.limbs[2 * TLC] << msbits_count) + (rs.limbs[2 * TLC - 1] >> (32 - msbits_count));
    icicle_math::template add_sub_limbs<2 * TLC + 2, true, false>(
      rs, get_mod_sub_for_storage_reduction(top_bits),
      rs); // subtracting the precomputed multiple of p from the look-up table
    // third and final step:
    storage<2 * TLC>& res = *reinterpret_cast<storage<2 * TLC>*>(rs.limbs);
    return reduce(Wide{res}); // finally, use barret reduction
  }

  HOST_DEVICE Derived& operator=(Derived const& other)
  {
#pragma unroll
    for (int i = 0; i < TLC; i++) {
      this->limbs_storage.limbs[i] = other.limbs_storage.limbs[i];
    }
    return *this;
  }

  friend HOST_DEVICE Derived operator*(const Derived& xs, const Derived& ys)
  {
    Wide xy = mul_wide(xs, ys); // full mult
    return reduce(xy);          // reduce mod p
  }

  friend HOST_DEVICE bool operator==(const Derived& xs, const Derived& ys)
  {
    return icicle_math::template is_equal<TLC>(xs.limbs_storage, ys.limbs_storage);
  }

  friend HOST_DEVICE bool operator!=(const Derived& xs, const Derived& ys) { return !(xs == ys); }

  template <const Derived& multiplier>
  static HOST_DEVICE_INLINE Derived mul_const(const Derived& xs)
  {
    constexpr bool is_u32 = []() {
      bool is_u32 = true;
      for (unsigned i = 1; i < TLC; i++)
        is_u32 &= (multiplier.limbs_storage.limbs[i] == 0);
      return is_u32;
    }();

    if constexpr (is_u32) return mul_unsigned<multiplier.limbs_storage.limbs[0], Derived>(xs);

    // This is not really a copy but required for CUDA compilation since the template param is not in the device memory
    Derived mult = multiplier;
    return mult * xs;
  }

  template <uint32_t multiplier, class T, unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE T mul_unsigned(const T& xs)
  {
    T rs = {};
    T temp = xs;
    bool is_zero = true;
#ifdef __CUDA_ARCH__
    UNROLL
#else
  #pragma unroll
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
  static constexpr HOST_DEVICE_INLINE Wide sqr_wide(const Derived& xs)
  {
    // TODO: change to a more efficient squaring
    return mul_wide<MODULUS_MULTIPLE>(xs, xs);
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Derived sqr(const Derived& xs)
  {
    // TODO: change to a more efficient squaring
    return xs * xs;
  }

  static constexpr HOST_DEVICE_INLINE Derived to_montgomery(const Derived& xs)
  {
    return xs * Derived{CONFIG::montgomery_r};
  }

  static constexpr HOST_DEVICE_INLINE Derived from_montgomery(const Derived& xs)
  {
    return xs * Derived{CONFIG::montgomery_r_inv};
  }

  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE Derived neg(const Derived& xs)
  {
    const ff_storage modulus = get_modulus<MODULUS_MULTIPLE>();
    Derived rs = {};
    sub_limbs<TLC, false>(modulus, xs.limbs_storage, rs.limbs_storage);
    return rs;
  }

  // Assumes the number is even!
  template <unsigned MODULUS_MULTIPLE = 1>
  static constexpr HOST_DEVICE_INLINE Derived div2(const Derived& xs)
  {
    Derived rs = {};
    icicle_math::template div2<TLC>(xs.limbs_storage, rs.limbs_storage);
    return sub_modulus<MODULUS_MULTIPLE>(rs);
  }

  static constexpr HOST_DEVICE_INLINE bool lt(const Derived& xs, const Derived& ys)
  {
    ff_storage dummy = {};
    uint32_t carry = sub_limbs<TLC, true>(xs.limbs_storage, ys.limbs_storage, dummy);
    return carry;
  }

  static constexpr HOST_DEVICE_INLINE bool is_odd(const Derived& xs) { return xs.limbs_storage.limbs[0] & 1; }

  static constexpr HOST_DEVICE_INLINE bool is_even(const Derived& xs) { return ~xs.limbs_storage.limbs[0] & 1; }

  static constexpr HOST_DEVICE Derived inverse(const Derived& xs)
  {
    if (xs == zero()) return zero();
    constexpr Derived one = Derived{CONFIG::one};
    constexpr Derived zero = Derived{CONFIG::zero};
    constexpr ff_storage modulus = CONFIG::modulus;
    Derived u = xs;
    Derived v = Derived{modulus};
    Derived b = one;
    Derived c = {};
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

#ifdef RING // only for rings, unnecessary for fields
      // Detect Non-Invertible Cases - rings only
      if (u == zero || v == zero) return zero; // If one side becomes 0, xs has no inverse
#endif
    }
    return (u == one) ? b : c;
  }

  static constexpr HOST_DEVICE Derived pow(Derived base, int exp)
  {
    Derived res = one();
    while (exp > 0) {
      if (exp & 1) res = res * base;
      base = base * base;
      exp >>= 1;
    }
    return res;
  }
};
