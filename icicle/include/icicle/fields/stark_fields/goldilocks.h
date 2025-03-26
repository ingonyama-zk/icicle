#pragma once

#include "icicle/math/storage.h"
#include "icicle/fields/field.h"
#include "icicle/fields/quartic_extension.h"
#include "icicle/fields/params_gen.h"

namespace goldilocks {

  template <class CONFIG>
  class GoldilocksField : public Field<CONFIG>
  {
    // using Wide = Field<CONFIG>::Wide

  public:
    static constexpr unsigned TLC = CONFIG::limbs_count;
    typedef storage<TLC> ff_storage;
    
    HOST_DEVICE_INLINE GoldilocksField(const GoldilocksField& other) : Field<CONFIG>(other) {}
    HOST_DEVICE_INLINE GoldilocksField(const uint32_t& x = 0) : Field<CONFIG>({x}) {}
    HOST_DEVICE_INLINE GoldilocksField(storage<CONFIG::limbs_count> x) : Field<CONFIG>{x} {}
    HOST_DEVICE_INLINE GoldilocksField(const Field<CONFIG>& other) : Field<CONFIG>(other) {}

    static constexpr HOST_DEVICE_INLINE GoldilocksField zero() { return GoldilocksField{CONFIG::zero}; }

    static constexpr HOST_DEVICE_INLINE GoldilocksField one() { return GoldilocksField{CONFIG::one}; }

    static constexpr HOST_DEVICE_INLINE GoldilocksField from(uint32_t value) { return GoldilocksField(value); }

    //TODO - The fact that for goldilocks the p_i's modulo p are {2^32-1, -2^32, 1, 2^32-1, -2^32, 1,...} can be used for an optimized version of the from functions.
    template <unsigned NLIMBS>
    static constexpr HOST_DEVICE_INLINE GoldilocksField from(const storage<NLIMBS>& xs) {
      return Field<CONFIG>::from(xs);
    }

    static constexpr HOST_DEVICE_INLINE GoldilocksField from(const std::byte* in, unsigned nof_bytes) {
      return Field<CONFIG>::from(in, nof_bytes);
    }    
    
    static HOST_INLINE GoldilocksField rand_host() { return GoldilocksField(Field<CONFIG>::rand_host()); }

    static void rand_host_many(GoldilocksField* out, int size) { 
      Field<CONFIG>::rand_host_many(static_cast<Field<CONFIG>*>(out), size); 
    }

    HOST_DEVICE_INLINE GoldilocksField& operator=(const Field<CONFIG>& other)
    {
      if (this != &other) { Field<CONFIG>::operator=(other); }
      return *this;
    }

    static constexpr HOST_DEVICE_INLINE GoldilocksField div2(const GoldilocksField& xs)
    {
      return Field<CONFIG>::div2(xs); //calls base sub_modulus or derived sub_modulus?
    }

    static constexpr HOST_DEVICE_INLINE GoldilocksField neg(const GoldilocksField& xs)
    {
      return Field<CONFIG>::neg(xs);
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE GoldilocksField reduce(typename Field<CONFIG>::Wide xs)
    {
      constexpr uint32_t gold_fact = uint32_t(-1);  //(1<<32) - 1
      GoldilocksField x_lo = {};
      x_lo.limbs_storage.limbs[0] = xs.limbs_storage.limbs[0];
      x_lo.limbs_storage.limbs[1] = xs.limbs_storage.limbs[1];
      // return x_lo;
      uint64_t temp = static_cast<uint64_t>(xs.limbs_storage.limbs[2]) * static_cast<uint64_t>(gold_fact);
      GoldilocksField x_hi_lo = {};
      x_hi_lo.limbs_storage.limbs[0] = temp & gold_fact;
      x_hi_lo.limbs_storage.limbs[1] = temp >> 32;
      GoldilocksField x_hi_hi = Field<CONFIG>::from(xs.limbs_storage.limbs[3]);
      return x_lo + x_hi_lo - x_hi_hi;
    }

    static constexpr HOST_DEVICE_INLINE GoldilocksField inverse(const GoldilocksField& x)
    {
      if (x == zero()) return zero();
      const GoldilocksField one = GoldilocksField{CONFIG::one};
      const GoldilocksField zero = GoldilocksField{CONFIG::zero};
      const ff_storage modulus = CONFIG::modulus;
      GoldilocksField u = x;
      GoldilocksField v = GoldilocksField{modulus};
      GoldilocksField b = one;
      GoldilocksField c = {};
      while (!(u == one) && !(v == one)) {
        while (Field<CONFIG>::is_even(u)) {
          uint32_t carry = 0;
          u = div2(u);
          if (Field<CONFIG>::is_odd(b)) carry = Field<CONFIG>::template add_limbs<TLC, true>(b.limbs_storage, modulus, b.limbs_storage);
          b = div2(b);
          if (carry) {
            b.limbs_storage.limbs[1] = b.limbs_storage.limbs[1] | (1U << 31);
          }
        }
        while (Field<CONFIG>::is_even(v)) {
          uint32_t carry = 0;
          v = div2(v);
          if (Field<CONFIG>::is_odd(c)) carry = Field<CONFIG>::template add_limbs<TLC, true>(c.limbs_storage, modulus, c.limbs_storage);
          c = div2(c);
          if (carry) {
            c.limbs_storage.limbs[1] = c.limbs_storage.limbs[1] | (1U << 31);
          }
        }
        if (Field<CONFIG>::lt(v, u)) {
          u = u - v;
          b = b - c;
        } else {
          v = v - u;
          c = c - b;
        }
      }
      return (u == one) ? b : c;
    }

    friend HOST_DEVICE_INLINE GoldilocksField operator+(GoldilocksField xs, const GoldilocksField& ys)
    {
      GoldilocksField rs = {};
      auto carry = Field<CONFIG>::template add_limbs<TLC, true>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
      if (carry){
        Field<CONFIG>::template add_limbs<TLC, false>(rs.limbs_storage, Field<CONFIG>::get_neg_modulus(), rs.limbs_storage);
        return rs;
      }
      return Field<CONFIG>::template sub_modulus<1>(rs);
    }

    friend HOST_DEVICE_INLINE GoldilocksField operator-(GoldilocksField xs, const GoldilocksField& ys)
    {
      Field<CONFIG> result = static_cast<const Field<CONFIG>&>(xs) - static_cast<const Field<CONFIG>&>(ys);
      return GoldilocksField{result};
    }

    friend HOST_DEVICE_INLINE GoldilocksField operator*(const GoldilocksField& xs, const GoldilocksField& ys)
    {
      typename Field<CONFIG>::Wide xy = Field<CONFIG>::mul_wide(xs, ys); 
      // return Field<CONFIG>::reduce(xy);
      return reduce(xy);
      // Field<CONFIG> result = static_cast<const Field<CONFIG>&>(xs) * static_cast<const Field<CONFIG>&>(ys);
      // return GoldilocksField{result};
    }

    static HOST_DEVICE_INLINE GoldilocksField inv_log_size(uint32_t logn){
      return Field<CONFIG>::inv_log_size(logn);
    }

    static constexpr HOST_DEVICE_INLINE GoldilocksField sqr(const GoldilocksField& xs) { return xs * xs; }

    static constexpr HOST_DEVICE_INLINE GoldilocksField to_montgomery(const GoldilocksField& xs) { return xs * GoldilocksField{CONFIG::montgomery_r}; }

    static constexpr HOST_DEVICE_INLINE GoldilocksField from_montgomery(const GoldilocksField& xs) { return xs * GoldilocksField{CONFIG::montgomery_r_inv}; }

    static constexpr HOST_DEVICE_INLINE GoldilocksField pow(GoldilocksField base, int exp)
    {
      return Field<CONFIG>::pow(base, exp);
    }
  };


  struct fp_config {
    static constexpr storage<2> modulus = {0x00000001, 0xffffffff};
    static constexpr unsigned reduced_digits_count = 9;
    static constexpr storage_array<reduced_digits_count, 2> reduced_digits = {
      {{0x00000001, 0x00000000},
      {0xffffffff, 0x00000000},
      {0x00000001, 0xfffffffe},
      {0x00000001, 0x00000000},
      {0xffffffff, 0x00000000},
      {0x00000001, 0xfffffffe},
      {0x00000001, 0x00000000},
      {0xffffffff, 0x00000000},
      {0x00000001, 0xfffffffe}}};
    PARAMS(modulus)
    MOD_SQR_SUBS()
    static constexpr storage_array<mod_subs_count, 2 * limbs_count + 2> mod_subs = {
      {{0x7fffffff, 0x00000001, 0xffffffff, 0x7fffffff, 0x00000000, 0x00000000},
      {0xffffffff, 0x00000001, 0xffffffff, 0xffffffff, 0x00000000, 0x00000000},
      {0x7fffffff, 0x00000002, 0xffffffff, 0x7fffffff, 0x00000001, 0x00000000},
      {0xffffffff, 0x00000002, 0xffffffff, 0xffffffff, 0x00000001, 0x00000000},
      {0x7fffffff, 0x00000003, 0xffffffff, 0x7fffffff, 0x00000002, 0x00000000},
      {0xffffffff, 0x00000003, 0xffffffff, 0xffffffff, 0x00000002, 0x00000000},
      {0x7fffffff, 0x00000004, 0xffffffff, 0x7fffffff, 0x00000003, 0x00000000},
      {0xffffffff, 0x00000004, 0xffffffff, 0xffffffff, 0x00000003, 0x00000000},
      {0x7fffffff, 0x00000005, 0xffffffff, 0x7fffffff, 0x00000004, 0x00000000},
      {0xffffffff, 0x00000005, 0xffffffff, 0xffffffff, 0x00000004, 0x00000000},
      {0x7fffffff, 0x00000006, 0xffffffff, 0x7fffffff, 0x00000005, 0x00000000},
      {0xffffffff, 0x00000006, 0xffffffff, 0xffffffff, 0x00000005, 0x00000000},
      {0x7fffffff, 0x00000007, 0xffffffff, 0x7fffffff, 0x00000006, 0x00000000},
      {0xffffffff, 0x00000007, 0xffffffff, 0xffffffff, 0x00000006, 0x00000000},
      {0x7fffffff, 0x00000008, 0xffffffff, 0x7fffffff, 0x00000007, 0x00000000},
      {0xffffffff, 0x00000008, 0xffffffff, 0xffffffff, 0x00000007, 0x00000000},
      {0x7fffffff, 0x00000009, 0xffffffff, 0x7fffffff, 0x00000008, 0x00000000}}};
    static constexpr storage<2> rou = {0x00000007, 0x00000000}; //todo - verify number
    TWIDDLES(modulus, rou)
  };

  /**
   * Scalar field. Is always a prime field.
   */
  typedef GoldilocksField<fp_config> scalar_t;

} // namespace goldilocks
