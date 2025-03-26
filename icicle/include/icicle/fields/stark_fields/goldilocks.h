#pragma once

#include "icicle/math/storage.h"
#include "icicle/fields/field.h"
#include "icicle/fields/quartic_extension.h"
#include "icicle/fields/params_gen.h"

//TODO - implement addition that deals with overflow

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

    /* This function receives a storage object (currently supports up to 576 bits) and reduces it to a field element
      between 0 and p. This is done using 2 steps:
      1. Splitting the number into TLC sized digits - xs = x_i * p_i = x_i * 2^(TLC*32*i).
      In the case of Mersenne, p_i modulo p turn out to be 2^i, therefore, we can replace the multiplication with
      shifts. At the end of this step the number is reduced from to 48 bits max (for a 576 input).
      2. The second step uses Mersenne reduction (splitting into digitd of 31 bits and adding) twice.*/
    template <unsigned NLIMBS>
    static constexpr HOST_DEVICE_INLINE GoldilocksField from(const storage<NLIMBS>& xs)
    {
      static_assert(NLIMBS * 32 <= 576); // for now we support up to 576 bits
      storage<2> rs = {};
      // first reduction step:
      for (int i = 0; i < NLIMBS; i++) {
        const GoldilocksField& xi =
          *reinterpret_cast<const GoldilocksField*>(xs.limbs + i); // use casting instead of copying
        storage<2> temp = {};
        temp.limbs[0] = xi.limbs_storage.limbs[0] << i; // in mersenne pi become shifts
        temp.limbs[1] = i ? xi.limbs_storage.limbs[0] >> (32 - i) : 0;
        icicle_math::template add_sub_limbs<2, false, false, true>(rs, temp, rs); // accumulation
      }
      // second reduction step:
      const uint32_t modulus = GoldilocksField::get_modulus().limbs[0];
      uint32_t tmp = ((rs.limbs[0] >> 31) | (rs.limbs[1] << 1)) +
                     (rs.limbs[0] & modulus); // mersenne reduction - max: 2^17 + 2^31-1 <= 2^32
      tmp = (tmp >> 31) + (tmp & modulus);    // max: 1 + 0 = 1
      return GoldilocksField{{tmp == modulus ? 0 : tmp}};
    }

    /* This is the non-template version of the from(storage) function above. It receives an array of bytes and its size
    and returns a field element after modular reduction. For now we support up to 576 bits. */
    static constexpr HOST_DEVICE_INLINE GoldilocksField from(const std::byte* in, unsigned nof_bytes)
    {
      storage<2> rs = {};
      unsigned constexpr bytes_per_field = 4;
      int size = nof_bytes / bytes_per_field;
      // first reduction step:
      for (int i = 0; i < size; i++) {
        const GoldilocksField& xi =
          *reinterpret_cast<const GoldilocksField*>(in + i * bytes_per_field); // use casting instead of copying
        storage<2> temp = {};
        temp.limbs[0] = xi.limbs_storage.limbs[0] << i; // in mersenne pi become shifts
        temp.limbs[1] = i ? xi.limbs_storage.limbs[0] >> (32 - i) : 0;
        icicle_math::template add_sub_limbs<2, false, false, true>(rs, temp, rs); // accumulation
      }
      // second reduction step:
      const uint32_t modulus = GoldilocksField::get_modulus().limbs[0];
      uint32_t tmp = ((rs.limbs[0] >> 31) | (rs.limbs[1] << 1)) +
                     (rs.limbs[0] & modulus); // mersenne reduction - max: 2^17 + 2^31-1 <= 2^32
      tmp = (tmp >> 31) + (tmp & modulus);    // max: 1 + 0 = 1
      return GoldilocksField{{tmp == modulus ? 0 : tmp}};
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

    HOST_DEVICE_INLINE uint32_t get_limb() const { return this->limbs_storage.limbs[0]; } //TODO - remove

    //  The `Wide` struct represents a redundant 32-bit form of the Mersenne Field.
    struct Wide { //TODO - remove
      uint32_t storage;
      static constexpr HOST_DEVICE_INLINE Wide from_field(const GoldilocksField& xs)
      {
        Wide out{};
        out.storage = xs.get_limb();
        return out;
      }
      static constexpr HOST_DEVICE_INLINE Wide from_number(const uint32_t& xs)
      {
        Wide out{};
        out.storage = xs;
        return out;
      }
      friend HOST_DEVICE_INLINE Wide operator+(Wide xs, const Wide& ys)
      {
        uint64_t tmp = (uint64_t)xs.storage + ys.storage;                   // max: 2^33 - 2 = 2^32(1) + (2^32 - 2)
        tmp = ((tmp >> 32) << 1) + (uint32_t)(tmp);                         // 2(1)+(2^32-2) = 2^32(1)+(0)
        return from_number((uint32_t)((tmp >> 32) << 1) + (uint32_t)(tmp)); // max: 2(1) + 0 = 2
      }
      friend HOST_DEVICE_INLINE Wide operator-(Wide xs, const Wide& ys)
      {
        uint64_t tmp = CONFIG::modulus_3 + xs.storage -
                       ys.storage; // max: 3(2^31-1) + 2^32-1 - 0 = 2^33 + 2^31-4 = 2^32(2) + (2^31-4)
        return from_number(((uint32_t)(tmp >> 32) << 1) + (uint32_t)(tmp)); // max: 2(2)+(2^31-4) = 2^31
      }
      template <unsigned MODULUS_MULTIPLE = 1>
      static constexpr HOST_DEVICE_INLINE Wide neg(const Wide& xs)
      {
        uint64_t tmp = CONFIG::modulus_3 - xs.storage;                      // max: 3(2^31-1) - 0 = 2^32(1) + (2^31 - 3)
        return from_number(((uint32_t)(tmp >> 32) << 1) + (uint32_t)(tmp)); // max: 2(1)+(2^31-3) = 2^31 - 1
      }
      friend HOST_DEVICE_INLINE Wide operator*(Wide xs, const Wide& ys)
      {
        uint64_t t1 = (uint64_t)xs.storage * ys.storage; // max: 2^64 - 2^33+1 = 2^32(2^32 - 2) + 1
        t1 = ((t1 >> 32) << 1) + (uint32_t)(t1);         // max: 2(2^32 - 2) + 1 = 2^32(1) + (2^32 - 3)
        return from_number((((uint32_t)(t1 >> 32)) << 1) + (uint32_t)(t1)); // max: 2(1) - (2^32 - 3) = 2^32 - 1
      }
    };

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
      // printf("am i here\n");
      constexpr uint32_t gold_fact = uint32_t(-1);  //(1<<32) - 1
      // GoldilocksField x_lo = GoldilocksField{xs.limbs_storage.limbs[0], xs.limbs_storage.limbs[1]};
      GoldilocksField x_lo = {};
      x_lo.limbs_storage.limbs[0] = xs.limbs_storage.limbs[0];
      x_lo.limbs_storage.limbs[1] = xs.limbs_storage.limbs[1];
      // return x_lo;
      uint64_t temp = static_cast<uint64_t>(xs.limbs_storage.limbs[2]) * static_cast<uint64_t>(gold_fact);
      // GoldilocksField x_hi_lo = GoldilocksField{temp & gold_fact, temp >> 32};
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

    // static constexpr HOST_DEVICE_INLINE Wide mul_wide(const GoldilocksField& xs, const GoldilocksField& ys)
    // {
    //   return Wide::from_field(xs) * Wide::from_field(ys);
    // }

    // template <unsigned MODULUS_MULTIPLE = 1>
    // static constexpr HOST_DEVICE_INLINE Wide sqr_wide(const GoldilocksField& xs)
    // {
    //   return mul_wide(xs, xs);
    // }

    static constexpr HOST_DEVICE_INLINE GoldilocksField sqr(const GoldilocksField& xs) { return xs * xs; }

    static constexpr HOST_DEVICE_INLINE GoldilocksField to_montgomery(const GoldilocksField& xs) { return xs; }

    static constexpr HOST_DEVICE_INLINE GoldilocksField from_montgomery(const GoldilocksField& xs) { return xs; }

    static constexpr HOST_DEVICE_INLINE GoldilocksField pow(GoldilocksField base, int exp)
    {
      GoldilocksField res = one();
      while (exp > 0) {
        if (exp & 1) res = res * base;
        base = base * base;
        exp >>= 1;
      }
      return res;
    }
  };


  struct fp_config {
    static constexpr storage<2> modulus = {0x00000001, 0xffffffff};
    PARAMS(modulus)

    static constexpr storage<2> rou = {0x00000010, 0x00000000}; //todo - verify number
    TWIDDLES(modulus, rou)

    // nonresidue to generate the extension field
    static constexpr uint32_t nonresidue = 3; //todo - verify number
    // true if nonresidue is negative.
    static constexpr bool nonresidue_is_negative = false; //todo - verify number
  };

  /**
   * Scalar field. Is always a prime field.
   */
  typedef GoldilocksField<fp_config> scalar_t;

  /**
   * Quartic extension field of `scalar_t` enabled if `-DEXT_FIELD` env variable is.
   */
  typedef QuarticExtensionField<fp_config, scalar_t> q_extension_t;

  /**
   * The default extension type
   */
  typedef q_extension_t extension_t;
} // namespace koalabear
