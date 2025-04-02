#pragma once

#include "icicle/math/storage.h"
#include "icicle/fields/field.h"
#include "icicle/fields/quartic_extension.h"
#include "icicle/fields/params_gen.h"

/*A few things to note about goldilocks field:
1. It has no slack bits (the modulus uses the entire 64 bits of storage) meaning we need to make sure there is no overflow in addition, and that the carry-less optimizations used in other fields do not apply here.
2. It has a special reduction algorithm due to the modulus special form.
3. In order to optimize addition and multiplication - the elements are reduced to the range 0<=x<2^64 instead of 0=<x<p. This requires changing the operator== implementation such that it will return true for values with a difference of p.
4. The operations that are implemented differently from the base Field class are: addition, reduction, inverse and comparison. The specific details are documented in the relevant fucntions. */
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
      return Field<CONFIG>::div2(xs);
    }

    static constexpr HOST_DEVICE_INLINE GoldilocksField neg(const GoldilocksField& xs)
    {
      return Field<CONFIG>::neg(xs);
    }

    /*This function implements the addition operation. Since the numbers are between 0 and 2^64 we either need to subtract p, 2p or do nothing.
    The case in which we need to subtract 2p is very rare and it happens only if both of the arguments are larger than p. Sometimes we can garantee that one of the arguments is smaller than p and then we use NO_OVERFLOW=true.
    That is the case in the call that is inside the reduction function. When we can't garantee this then we hint the compiler that this is a rare case.*/
     template <bool NO_OVERFLOW = false>
    static HOST_DEVICE_INLINE GoldilocksField goldi_add(const GoldilocksField& xs, const GoldilocksField& ys){
      GoldilocksField rs = {};
      const ff_storage modulus = Field<CONFIG>::get_modulus();
      if constexpr (NO_OVERFLOW == false) { // Handle the rare case where we would need to subtract 2p
        if (__builtin_expect(xs.limbs_storage.limbs64[0] >= modulus.limbs64[0], 0)) rs.limbs_storage.limbs64[0] = xs.limbs_storage.limbs64[0] - modulus.limbs64[0]; // It is actually more efficient to check only one of the arguments.
        else rs.limbs_storage.limbs64[0] = xs.limbs_storage.limbs64[0];
      }
      else {
        rs.limbs_storage.limbs64[0] = xs.limbs_storage.limbs64[0];
      }
      auto carry = Field<CONFIG>::template add_limbs<TLC, true>(rs.limbs_storage, ys.limbs_storage, rs.limbs_storage); // Do the addition
      if (carry){
          Field<CONFIG>::template add_limbs<TLC, false>(rs.limbs_storage, Field<CONFIG>::get_neg_modulus(), rs.limbs_storage); // Adding (-p) effectively sutracts p in case there is a carry. This is garanteed no to overflow since we already took care of the rare case.
      }
      return rs;
    }

    friend HOST_DEVICE_INLINE GoldilocksField operator+(const GoldilocksField& xs, const GoldilocksField& ys)
    {
      return goldi_add(xs, ys);
    }

    friend HOST_DEVICE_INLINE GoldilocksField operator-(GoldilocksField xs, const GoldilocksField& ys)
    {
      Field<CONFIG> result = static_cast<const Field<CONFIG>&>(xs) - static_cast<const Field<CONFIG>&>(ys);
      return GoldilocksField{result};
    }

    /*This function performs the goldilocks reduction:
    xs[63:0] + xs[95:64] * (2^32 - 1) - xs[127:96]
    First it does the subtraction - xs[63:0] - xs[127:96] and hints the compiler that it is rare that xs[63:0] < xs[127:96].
    Then it adds xs[95:64] * (2^32 - 1) which is garanteed to be smaller than p - that's why we use the addition with NO_OVERFLOW=true*/
    static constexpr HOST_DEVICE_INLINE GoldilocksField reduce(const typename Field<CONFIG>::Wide xs)
    {
      constexpr uint32_t gold_fact = uint32_t(-1);  //(2^32 - 1)
      GoldilocksField rs = {};
      const GoldilocksField& x_lo = *reinterpret_cast<const GoldilocksField*>(xs.limbs_storage.limbs);
      GoldilocksField x_hi_hi = Field<CONFIG>::from(xs.limbs_storage.limbs[3]);
      auto carry = Field<CONFIG>::template sub_limbs<TLC, true>(x_lo.limbs_storage, x_hi_hi.limbs_storage, rs.limbs_storage); // xs[63:0] - xs[127:96]
      if (__builtin_expect(carry, 0)) {
        Field<CONFIG>::template sub_limbs<TLC, false>(rs.limbs_storage, Field<CONFIG>::get_neg_modulus(), rs.limbs_storage); // cannot underflow
      }
      GoldilocksField x_hi_lo = {};
      x_hi_lo.limbs_storage.limbs64[0] = static_cast<uint64_t>(xs.limbs_storage.limbs[2]) * static_cast<uint64_t>(gold_fact); // xs[95:64] * (2^32 - 1)
      return goldi_add<true>(rs,x_hi_lo); // NO_OVERFLOW=true
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
            b.limbs_storage.limbs[1] = b.limbs_storage.limbs[1] | (1U << 31); // If there is a carry then after the division by 2 we can insert it as the top bit
          }
        }
        while (Field<CONFIG>::is_even(v)) {
          uint32_t carry = 0;
          v = div2(v);
          if (Field<CONFIG>::is_odd(c)) carry = Field<CONFIG>::template add_limbs<TLC, true>(c.limbs_storage, modulus, c.limbs_storage);
          c = div2(c);
          if (carry) {
            c.limbs_storage.limbs[1] = c.limbs_storage.limbs[1] | (1U << 31); // If there is a carry then after the division by 2 we can insert it as the top bit
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

    /*Since we allow the elements to be between 0 and 2^64, if they are larger than p we need to subtract p before the comparison. This is a rare case so we hint the compiler.*/
    friend HOST_DEVICE bool operator==(const GoldilocksField& xs, const GoldilocksField& ys)
    {
      const ff_storage modulus = Field<CONFIG>::get_modulus();
      GoldilocksField xr = {};
      if (__builtin_expect(xs.limbs_storage.limbs64[0] >= modulus.limbs64[0], 0)) xr.limbs_storage.limbs64[0] = xs.limbs_storage.limbs64[0] - modulus.limbs64[0];
      else xr.limbs_storage.limbs64[0] = xs.limbs_storage.limbs64[0];
      GoldilocksField yr = {};
      if (__builtin_expect(ys.limbs_storage.limbs64[0] >= modulus.limbs64[0], 0)) yr.limbs_storage.limbs64[0] = ys.limbs_storage.limbs64[0] - modulus.limbs64[0];
      else yr.limbs_storage.limbs64[0] = ys.limbs_storage.limbs64[0];
      return icicle_math::template is_equal<TLC>(xr.limbs_storage, yr.limbs_storage);
    }

    friend HOST_DEVICE bool operator!=(const GoldilocksField& xs, const GoldilocksField& ys) { return !(xs == ys); }


    static HOST_INLINE GoldilocksField omega(uint32_t logn)
    {
      if (logn == 0) { return GoldilocksField{CONFIG::one}; }
  
      if (logn > CONFIG::omegas_count) {
        THROW_ICICLE_ERR(icicle::eIcicleError::INVALID_ARGUMENT, "ModArith: Invalid omega index");
      }
  
      GoldilocksField omega = GoldilocksField{CONFIG::rou};
      for (int i = 0; i < CONFIG::omegas_count - logn; i++){
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
    static constexpr storage<2> rou = {0xda58878c, 0x185629dc};
    TWIDDLES(modulus, rou)
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