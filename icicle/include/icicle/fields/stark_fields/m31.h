#pragma once

#include "icicle/math/storage.h"
#include "icicle/math/modular_arithmetic.h"
#include "icicle/fields/complex_extension.h"
#include "icicle/fields/quartic_extension.h"
#include "icicle/fields/params_gen.h"

#include <iostream>
#include <cstddef>
#include <iomanip>

namespace m31 {
  template <class CONFIG>
  class MersenneField : public ModArith<MersenneField<CONFIG>, CONFIG>
  {
    using Base = ModArith<MersenneField<CONFIG>, CONFIG>;

  public:
    // Copy-assignment forwards to ModArith implementation.
    HOST_DEVICE MersenneField& operator=(const MersenneField& other)
    {
      ModArith<MersenneField<CONFIG>, CONFIG>::operator=(other);
      return *this;
    }

    HOST_DEVICE_INLINE MersenneField(const MersenneField& other) : Base(other) {}
    HOST_DEVICE_INLINE MersenneField(const uint32_t& x = 0) : Base({x}) {}
    HOST_DEVICE_INLINE MersenneField(storage<CONFIG::limbs_count> x) : Base{x} {}
    HOST_DEVICE_INLINE MersenneField(const Field<CONFIG>& other) : Base(other.limbs_storage) {}

    static constexpr HOST_DEVICE_INLINE MersenneField zero() { return MersenneField{CONFIG::zero}; }

    static constexpr HOST_DEVICE_INLINE MersenneField one() { return MersenneField{CONFIG::one}; }

    static constexpr HOST_DEVICE_INLINE MersenneField from(uint32_t value) { return Base::from(value); }

    /* This function receives a storage object (currently supports up to 576 bits) and reduces it to a field element
      between 0 and p. This is done using 2 steps:
      1. Splitting the number into TLC sized digits - xs = x_i * p_i = x_i * 2^(TLC*32*i).
      In the case of Mersenne, p_i modulo p turn out to be 2^i, therefore, we can replace the multiplication with
      shifts. At the end of this step the number is reduced from to 48 bits max (for a 576 input).
      2. The second step uses Mersenne reduction (splitting into digitd of 31 bits and adding) twice.*/
    template <unsigned NLIMBS>
    static constexpr HOST_DEVICE_INLINE MersenneField from(const storage<NLIMBS>& xs)
    {
      static_assert(NLIMBS * 32 <= 576); // for now we support up to 576 bits
      storage<2> rs = {};
      // first reduction step:
      for (int i = 0; i < NLIMBS; i++) {
        const MersenneField& xi =
          *reinterpret_cast<const MersenneField*>(xs.limbs + i); // use casting instead of copying
        storage<2> temp = {};
        temp.limbs[0] = xi.limbs_storage.limbs[0] << i; // in mersenne pi become shifts
        temp.limbs[1] = i ? xi.limbs_storage.limbs[0] >> (32 - i) : 0;
        icicle_math::template add_sub_limbs<2, false, false, true>(rs, temp, rs); // accumulation
      }
      // second reduction step:
      const uint32_t modulus = MersenneField::get_modulus().limbs[0];
      uint32_t tmp = ((rs.limbs[0] >> 31) | (rs.limbs[1] << 1)) +
                     (rs.limbs[0] & modulus); // mersenne reduction - max: 2^17 + 2^31-1 <= 2^32
      tmp = (tmp >> 31) + (tmp & modulus);    // max: 1 + 0 = 1
      return MersenneField{{tmp == modulus ? 0 : tmp}};
    }

    /* This is the non-template version of the from(storage) function above. It receives an array of bytes and its size
    and returns a field element after modular reduction. For now we support up to 576 bits. */
    static constexpr HOST_DEVICE_INLINE MersenneField from(const std::byte* in, unsigned nof_bytes)
    {
      storage<2> rs = {};
      unsigned constexpr bytes_per_field = 4;
      int size = nof_bytes / bytes_per_field;
      // first reduction step:
      for (int i = 0; i < size; i++) {
        const MersenneField& xi =
          *reinterpret_cast<const MersenneField*>(in + i * bytes_per_field); // use casting instead of copying
        storage<2> temp = {};
        temp.limbs[0] = xi.limbs_storage.limbs[0] << i; // in mersenne pi become shifts
        temp.limbs[1] = i ? xi.limbs_storage.limbs[0] >> (32 - i) : 0;
        icicle_math::template add_sub_limbs<2, false, false, true>(rs, temp, rs); // accumulation
      }
      // second reduction step:
      const uint32_t modulus = MersenneField::get_modulus().limbs[0];
      uint32_t tmp = ((rs.limbs[0] >> 31) | (rs.limbs[1] << 1)) +
                     (rs.limbs[0] & modulus); // mersenne reduction - max: 2^17 + 2^31-1 <= 2^32
      tmp = (tmp >> 31) + (tmp & modulus);    // max: 1 + 0 = 1
      return MersenneField{{tmp == modulus ? 0 : tmp}};
    }

    static HOST_INLINE MersenneField rand_host()
    {
      Base field_val = Base::rand_host();
      return MersenneField{{field_val.limbs_storage}};
    }
    static void rand_host_many(MersenneField* out, int size)
    {
      for (int i = 0; i < size; i++)
        out[i] = rand_host();
    }

    HOST_DEVICE_INLINE MersenneField& operator=(const Field<CONFIG>& other)
    {
      if (this != &other) { Base::operator=(other); }
      return *this;
    }

    HOST_DEVICE_INLINE uint32_t get_limb() const { return this->limbs_storage.limbs[0]; }

    //  The `Wide` struct represents a redundant 32-bit form of the Mersenne Field.
    struct Wide {
      uint32_t storage;
      static constexpr HOST_DEVICE_INLINE Wide from_field(const MersenneField& xs)
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

      HOST_DEVICE_INLINE Wide operator+(const Wide& ys) const
      {
        uint64_t tmp = (uint64_t)storage + ys.storage;                      // max: 2^33 - 2 = 2^32(1) + (2^32 - 2)
        tmp = ((tmp >> 32) << 1) + (uint32_t)(tmp);                         // 2(1)+(2^32-2) = 2^32(1)+(0)
        return from_number((uint32_t)((tmp >> 32) << 1) + (uint32_t)(tmp)); // max: 2(1) + 0 = 2
      }

      HOST_DEVICE_INLINE Wide operator-(const Wide& ys) const
      {
        uint64_t tmp =
          CONFIG::modulus_3 + storage - ys.storage; // max: 3(2^31-1) + 2^32-1 - 0 = 2^33 + 2^31-4 = 2^32(2) + (2^31-4)
        return from_number(((uint32_t)(tmp >> 32) << 1) + (uint32_t)(tmp)); // max: 2(2)+(2^31-4) = 2^31
      }

      constexpr HOST_DEVICE_INLINE Wide neg() const
      {
        uint64_t tmp = CONFIG::modulus_3 - storage;                         // max: 3(2^31-1) - 0 = 2^32(1) + (2^31 - 3)
        return from_number(((uint32_t)(tmp >> 32) << 1) + (uint32_t)(tmp)); // max: 2(1)+(2^31-3) = 2^31 - 1
      }

      HOST_DEVICE_INLINE Wide operator*(const Wide& ys) const
      {
        uint64_t t1 = (uint64_t)storage * ys.storage; // max: 2^64 - 2^33+1 = 2^32(2^32 - 2) + 1
        t1 = ((t1 >> 32) << 1) + (uint32_t)(t1);      // max: 2(2^32 - 2) + 1 = 2^32(1) + (2^32 - 3)
        return from_number((((uint32_t)(t1 >> 32)) << 1) + (uint32_t)(t1)); // max: 2(1) - (2^32 - 3) = 2^32 - 1
      }

      constexpr HOST_DEVICE_INLINE MersenneField reduce() const { return MersenneField::reduce(*this); }
    };

    constexpr HOST_DEVICE_INLINE MersenneField div2(const uint32_t& power = 1) const
    {
      uint32_t t = this->get_limb();
      return MersenneField{{((t >> power) | (t << (31 - power))) & MersenneField::get_modulus().limbs[0]}};
    }

    constexpr HOST_DEVICE_INLINE MersenneField neg() const
    {
      uint32_t t = this->get_limb();
      return MersenneField{{t == 0 ? t : MersenneField::get_modulus().limbs[0] - t}};
    }
    template <unsigned LIMBS_COUNT = 1>
    constexpr HOST_DEVICE_INLINE MersenneField reduce() const
    {
      const uint32_t modulus = MersenneField::get_modulus().limbs[0];
      uint32_t tmp =
        ((uint64_t)this->get_limb() >> 31) + ((uint64_t)this->get_limb() & modulus); // max: 1 + 2^31-1 = 2^31
      tmp = (tmp >> 31) + (tmp & modulus);                                           // max: 1 + 0 = 1
      return MersenneField{{tmp == modulus ? 0 : tmp}};
    }

    constexpr HOST_DEVICE_INLINE MersenneField inverse() const
    {
      uint32_t xs = this->get_limb();
      if (xs <= 1) return MersenneField{{xs}};
      uint32_t a = 1, b = 0, y = xs, z = MersenneField::get_modulus().limbs[0], m = z;
      while (1) {
#ifdef __CUDA_ARCH__
        uint32_t e = __ffs(y) - 1;
#else
        uint32_t e = __builtin_ctz(y);
#endif
        y >>= e;
        if (a >= m) {
          a = (a & m) + (a >> 31);
          if (a == m) a = 0;
        }
        a = ((a >> e) | (a << (31 - e))) & m;
        if (y == 1) return MersenneField{{a}};
        e = a + b;
        b = a;
        a = e;
        e = y + z;
        z = y;
        y = e;
      }
    }

    HOST_DEVICE_INLINE MersenneField operator+(const MersenneField& ys) const
    {
      uint32_t m = MersenneField::get_modulus().limbs[0];
      uint32_t t = get_limb() + ys.get_limb();
      if (t > m) t = (t & m) + (t >> 31);
      if (t == m) t = 0;
      return MersenneField{{t}};
    }

    HOST_DEVICE_INLINE MersenneField operator-(const MersenneField& ys) const { return *this + ys.neg(); }

    HOST_DEVICE_INLINE MersenneField operator*(const MersenneField& ys) const
    {
      uint64_t x = (uint64_t)(get_limb())*ys.get_limb();
      uint32_t t = ((x >> 31) + (x & MersenneField::get_modulus().limbs[0]));
      uint32_t m = MersenneField::get_modulus().limbs[0];
      if (t > m) t = (t & m) + (t >> 31);
      if (t > m) t = (t & m) + (t >> 31);
      if (t == m) t = 0;
      return MersenneField{{t}};
    }

    constexpr HOST_DEVICE_INLINE Wide mul_wide(const MersenneField& ys) const
    {
      return Wide::from_field(*this) * Wide::from_field(ys);
    }

    constexpr HOST_DEVICE_INLINE Wide sqr_wide() const { return mul_wide(*this); }

    constexpr HOST_DEVICE_INLINE MersenneField to_montgomery() const { return *this; }

    HOST_DEVICE_INLINE MersenneField from_montgomery() const { return *this; }

    HOST_DEVICE_INLINE MersenneField pow(int exp) const
    {
      MersenneField res = one();
      MersenneField base = *this;
      while (exp > 0) {
        if (exp & 1) res = res * base;
        base = base * base;
        exp >>= 1;
      }
      return res;
    }

    // Add a static reduce method that accepts the Wide type
    static constexpr HOST_DEVICE_INLINE MersenneField reduce(const Wide& wide)
    {
      const uint32_t modulus = MersenneField::get_modulus().limbs[0];
      uint32_t tmp = ((uint64_t)wide.storage >> 31) + ((uint64_t)wide.storage & modulus); // max: 1 + 2^31-1 = 2^31
      tmp = (tmp >> 31) + (tmp & modulus);                                                // max: 1 + 0 = 1
      return MersenneField{{tmp == modulus ? 0 : tmp}};
    }
  };
  struct fp_config {
    static constexpr unsigned limbs_count = 1;
    static constexpr unsigned omegas_count = 1;
    static constexpr unsigned modulus_bit_count = 31;
    static constexpr unsigned num_of_reductions = 1;

    static constexpr storage<limbs_count> modulus = {0x7fffffff};
    static constexpr storage<limbs_count> modulus_2 = {0xfffffffe};
    static constexpr uint64_t modulus_3 = 0x17ffffffd;
    static constexpr storage<limbs_count> modulus_4 = {0xfffffffc};
    static constexpr storage<limbs_count> neg_modulus = {0x87ffffff};
    static constexpr storage<2 * limbs_count> modulus_wide = {0x7fffffff, 0x00000000};
    static constexpr storage<2 * limbs_count> modulus_squared = {0x00000001, 0x3fffffff};
    static constexpr storage<2 * limbs_count> modulus_squared_2 = {0x00000002, 0x7ffffffe};
    static constexpr storage<2 * limbs_count> modulus_squared_4 = {0x00000004, 0xfffffffc};

    static constexpr storage<limbs_count> m = {0x80000001};
    static constexpr storage<limbs_count> one = {0x00000001};
    static constexpr storage<limbs_count> zero = {0x00000000};
    static constexpr storage<limbs_count> montgomery_r = {0x00000001};
    static constexpr storage<limbs_count> montgomery_r_inv = {0x00000001};

    static constexpr storage_array<omegas_count, limbs_count> omega = {{{0x7ffffffe}}};

    static constexpr storage_array<omegas_count, limbs_count> omega_inv = {{{0x7ffffffe}}};

    static constexpr storage_array<omegas_count, limbs_count> inv = {{{0x40000000}}};
    // nonresidue to generate the extension field
    static constexpr uint32_t nonresidue = 1;
    // true if nonresidue is negative.
    static constexpr bool nonresidue_is_negative = true;
  };

  /**
   * Scalar field. Is always a prime field.
   */
  typedef MersenneField<fp_config> scalar_t;

  /**
   * Quartic extension field of `scalar_t` enabled if `-DEXT_FIELD` env variable is.
   */
  typedef ComplexExtensionField<fp_config, scalar_t> c_extension_t;

  /**
   * Quartic extension field of `scalar_t` enabled if `-DEXT_FIELD` env variable is.
   */
  typedef QuarticExtensionField<fp_config, scalar_t> q_extension_t;

  /**
   * The default extension type
   */
  typedef q_extension_t extension_t;
} // namespace m31