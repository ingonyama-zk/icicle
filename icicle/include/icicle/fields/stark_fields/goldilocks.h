#pragma once

#include "icicle/math/storage.h"
#include "icicle/fields/field.h"
#include "icicle/fields/quartic_extension.h"
#include "icicle/fields/params_gen.h"

/*A few things to note about goldilocks field:
1. It has no slack bits (the modulus uses the entire 64 bits of storage) meaning we need to make sure there is no
overflow in addition, and that the carry-less optimizations used in other fields do not apply here.
2. It has a special reduction algorithm due to the modulus special form.
3. The operations that are implemented differently from the base Field class are: addition, reduction, inverse and
comparison. The specific details are documented in the relevant functions. */
namespace goldilocks {

  template <class CONFIG>
  class GoldilocksField : public Field<CONFIG>
  {
  public:
    static constexpr unsigned TLC = CONFIG::limbs_count;
    typedef storage<TLC> ff_storage;

    HOST_DEVICE_INLINE GoldilocksField(const GoldilocksField& other) : Field<CONFIG>(other) {}
    HOST_DEVICE_INLINE GoldilocksField(const uint32_t& x = 0) : Field<CONFIG>({x}) {}
    HOST_DEVICE_INLINE GoldilocksField(storage<CONFIG::limbs_count> x) : Field<CONFIG>{x} {}
    HOST_DEVICE_INLINE GoldilocksField(const Field<CONFIG>& other) : Field<CONFIG>(other) {}

    static constexpr HOST_DEVICE_INLINE GoldilocksField zero() { return GoldilocksField{CONFIG::zero}; }

    static constexpr HOST_DEVICE_INLINE GoldilocksField one() { return GoldilocksField{CONFIG::one}; }

    static constexpr HOST_DEVICE_INLINE GoldilocksField from(uint32_t value) { return Field<CONFIG>::from(value); }

    // TODO - The fact that for goldilocks the p_i's modulo p are {2^32-1, -2^32, 1, 2^32-1, -2^32, 1,...} can be used
    // for an optimized version of the from functions.
    template <unsigned NLIMBS>
    static constexpr HOST_DEVICE_INLINE GoldilocksField from(const storage<NLIMBS>& xs)
    {
      return Field<CONFIG>::from(xs);
    }

    static constexpr HOST_DEVICE_INLINE GoldilocksField from(const std::byte* in, unsigned nof_bytes)
    {
      return Field<CONFIG>::from(in, nof_bytes);
    }

    static HOST_INLINE GoldilocksField rand_host() { return GoldilocksField(Field<CONFIG>::rand_host()); }

    static void rand_host_many(GoldilocksField* out, int size)
    {
      Field<CONFIG>::rand_host_many(static_cast<Field<CONFIG>*>(out), size);
    }

    HOST_DEVICE_INLINE GoldilocksField& operator=(const Field<CONFIG>& other)
    {
      if (this != &other) { Field<CONFIG>::operator=(other); }
      return *this;
    }

    static constexpr HOST_DEVICE_INLINE GoldilocksField div2(const GoldilocksField& xs)
    {
      return Field<CONFIG>::div2(xs);
    }

    static constexpr HOST_DEVICE_INLINE GoldilocksField neg(const GoldilocksField& xs)
    {
      return Field<CONFIG>::neg(xs);
    }

    friend HOST_DEVICE_INLINE GoldilocksField operator+(const GoldilocksField& xs, const GoldilocksField& ys)
    {
      GoldilocksField rs = {};
      icicle_math::goldi_add(xs.limbs_storage, ys.limbs_storage, Field<CONFIG>::get_modulus(), Field<CONFIG>::get_neg_modulus(), rs.limbs_storage);
      return rs;
    }

    friend HOST_DEVICE_INLINE GoldilocksField operator-(GoldilocksField xs, const GoldilocksField& ys)
    {
      Field<CONFIG> result = static_cast<const Field<CONFIG>&>(xs) - static_cast<const Field<CONFIG>&>(ys);
      return GoldilocksField{result};
    }

    /*This function performs the goldilocks reduction:
    xs[63:0] + xs[95:64] * (2^32 - 1) - xs[127:96]
     */
    static constexpr HOST_DEVICE_INLINE GoldilocksField reduce(const typename Field<CONFIG>::Wide xs)
    {
      GoldilocksField rs = {};
      icicle_math::goldi_reduce(xs.limbs_storage, Field<CONFIG>::get_modulus(), Field<CONFIG>::get_neg_modulus(), rs.limbs_storage);
      return rs;
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
          if (Field<CONFIG>::is_odd(b))
            carry = Field<CONFIG>::template add_limbs<TLC, true>(b.limbs_storage, modulus, b.limbs_storage);
          b = div2(b);
          if (carry) {
            b.limbs_storage.limbs[1] =
              b.limbs_storage.limbs[1] |
              (1U << 31); // If there is a carry then after the division by 2 we can insert it as the top bit
          }
        }
        while (Field<CONFIG>::is_even(v)) {
          uint32_t carry = 0;
          v = div2(v);
          if (Field<CONFIG>::is_odd(c))
            carry = Field<CONFIG>::template add_limbs<TLC, true>(c.limbs_storage, modulus, c.limbs_storage);
          c = div2(c);
          if (carry) {
            c.limbs_storage.limbs[1] =
              c.limbs_storage.limbs[1] |
              (1U << 31); // If there is a carry then after the division by 2 we can insert it as the top bit
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

    friend HOST_DEVICE_INLINE GoldilocksField operator*(const GoldilocksField& xs, const GoldilocksField& ys)
    {
      typename Field<CONFIG>::Wide xy = Field<CONFIG>::mul_wide(xs, ys);
      return reduce(xy);
    }

    static HOST_INLINE GoldilocksField omega(uint32_t logn)
    {
      if (logn == 0) { return GoldilocksField{CONFIG::one}; }

      if (logn > CONFIG::omegas_count) {
        THROW_ICICLE_ERR(icicle::eIcicleError::INVALID_ARGUMENT, "ModArith: Invalid omega index");
      }

      GoldilocksField omega = GoldilocksField{CONFIG::rou};
      for (int i = 0; i < CONFIG::omegas_count - logn; i++) {
        omega = sqr(omega);
      }
      return omega;
    }

    static HOST_INLINE GoldilocksField omega_inv(uint32_t logn)
    {
      if (logn == 0) { return GoldilocksField{CONFIG::one}; }

      if (logn > CONFIG::omegas_count) {
        THROW_ICICLE_ERR(icicle::eIcicleError::INVALID_ARGUMENT, "ModArith: Invalid omega_inv index");
      }

      GoldilocksField omega = inverse(GoldilocksField{CONFIG::rou});
      for (int i = 0; i < CONFIG::omegas_count - logn; i++)
        omega = sqr(omega);
      return omega;
    }

    static HOST_DEVICE_INLINE GoldilocksField inv_log_size(uint32_t logn) { return Field<CONFIG>::inv_log_size(logn); }

    static constexpr HOST_DEVICE_INLINE GoldilocksField sqr(const GoldilocksField& xs) { return xs * xs; }

    static constexpr HOST_DEVICE_INLINE GoldilocksField to_montgomery(const GoldilocksField& xs)
    {
      GoldilocksField t = GoldilocksField{CONFIG::montgomery_r};

      return xs * GoldilocksField{CONFIG::montgomery_r};
    }

    static constexpr HOST_DEVICE_INLINE GoldilocksField from_montgomery(const GoldilocksField& xs)
    {
      return xs * GoldilocksField{CONFIG::montgomery_r_inv};
    }

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
    static constexpr storage<2> modulus = {0x00000001, 0xffffffff}; // 2^64 - 2^32 + 1
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
    static constexpr unsigned limbs_count = 2;
    static constexpr unsigned modulus_bit_count = 64;
    static constexpr storage<limbs_count> zero = {};
    static constexpr storage<limbs_count> one = {1};
    static constexpr storage<limbs_count> neg_modulus = {0xffffffff, 0x00000000};
    static constexpr storage<limbs_count> montgomery_r = {0xffffffff, 0x00000000};
    static constexpr storage<limbs_count> montgomery_r_inv = {0x00000001, 0xfffffffe};
    // The next 4 parameters are unused and are only needed for compilation:
    static constexpr storage<limbs_count> modulus_2 = {0x00000000, 0x00000000};
    static constexpr storage<limbs_count> modulus_4 = {0x00000000, 0x00000000};
    static constexpr storage<limbs_count> m = {0x00000000, 0x00000000};
    static constexpr unsigned num_of_reductions = 0;
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
    static constexpr storage<2> rou = {0xda58878c, 0x185629dc};
    static constexpr unsigned omegas_count = 32;
    static constexpr storage_array<omegas_count, limbs_count> inv = {
      {{0x80000001, 0x7fffffff}, {0x40000001, 0xbfffffff}, {0x20000001, 0xdfffffff}, {0x10000001, 0xefffffff},
       {0x08000001, 0xf7ffffff}, {0x04000001, 0xfbffffff}, {0x02000001, 0xfdffffff}, {0x01000001, 0xfeffffff},
       {0x00800001, 0xff7fffff}, {0x00400001, 0xffbfffff}, {0x00200001, 0xffdfffff}, {0x00100001, 0xffefffff},
       {0x00080001, 0xfff7ffff}, {0x00040001, 0xfffbffff}, {0x00020001, 0xfffdffff}, {0x00010001, 0xfffeffff},
       {0x00008001, 0xffff7fff}, {0x00004001, 0xffffbfff}, {0x00002001, 0xffffdfff}, {0x00001001, 0xffffefff},
       {0x00000801, 0xfffff7ff}, {0x00000401, 0xfffffbff}, {0x00000201, 0xfffffdff}, {0x00000101, 0xfffffeff},
       {0x00000081, 0xffffff7f}, {0x00000041, 0xffffffbf}, {0x00000021, 0xffffffdf}, {0x00000011, 0xffffffef},
       {0x00000009, 0xfffffff7}, {0x00000005, 0xfffffffb}, {0x00000003, 0xfffffffd}, {0x00000002, 0xfffffffe}}};
  };

  /**
   * Scalar field. Is always a prime field.
   */
  typedef GoldilocksField<fp_config> scalar_t;

} // namespace goldilocks

template <class CONFIG>
struct std::hash<goldilocks::GoldilocksField<CONFIG>> {
  std::size_t operator()(const goldilocks::GoldilocksField<CONFIG>& key) const
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
struct SharedMemory<goldilocks::GoldilocksField<CONFIG>> {
  __device__ goldilocks::GoldilocksField<CONFIG>* getPointer()
  {
    extern __shared__ goldilocks::GoldilocksField<CONFIG> s_scalar_[];
    return s_scalar_;
  }
};

#endif // __CUDACC__