#pragma once

#include "icicle/math/storage.h"
#include "icicle/fields/field.h"
#include "icicle/fields/params_gen.h"
#include "icicle/fields/complex_extension.h"
#ifdef __CUDACC__
  #include "goldilocks_cuda_math.h"
  #include "gpu-utils/sharedmem.h"
#endif // __CUDACC__
#include "icicle/math/goldilocks_host_math.h"

/*A few things to note about goldilocks field:
1. It has no slack bits (the modulus uses the entire 64 bits of storage) meaning we need to make sure there is no
overflow in addition, and that the carry-less optimizations used in other fields do not apply here.
2. It has a special reduction algorithm due to the modulus special form.
3. The operations that are implemented differently from the base Field class are: addition, reduction, inverse and
comparison. The specific details are documented in the relevant functions. */
namespace goldilocks {

  template <class CONFIG>
  class GoldilocksField : public ModArith<GoldilocksField<CONFIG>, CONFIG>
  {
    using Base = ModArith<GoldilocksField<CONFIG>, CONFIG>;

  public:
    static constexpr unsigned TLC = CONFIG::limbs_count;
    static constexpr unsigned NBITS = CONFIG::modulus_bit_count;
    typedef storage<TLC> ff_storage;

    HOST_DEVICE_INLINE GoldilocksField(const GoldilocksField& other) : Base(other) {}
    HOST_DEVICE_INLINE GoldilocksField(const uint32_t& x = 0) : Base({x}) {}
    HOST_DEVICE_INLINE GoldilocksField(storage<CONFIG::limbs_count> x) : Base{x} {}
    HOST_DEVICE_INLINE GoldilocksField(const Field<CONFIG>& other) : Base(other.limbs_storage) {}

    static constexpr HOST_DEVICE_INLINE GoldilocksField zero() { return GoldilocksField{CONFIG::zero}; }

    static constexpr HOST_DEVICE_INLINE GoldilocksField one() { return GoldilocksField{CONFIG::one}; }

    static constexpr HOST_DEVICE_INLINE GoldilocksField from(uint32_t value) { return Base::from(value); }

    // TODO - The fact that for goldilocks the p_i's modulo p are {2^32-1, -2^32, 1, 2^32-1, -2^32, 1,...} can be used
    // for an optimized version of the from functions.
    // The implementation of the following 2 functions is exactly the same as in modular_arithmetic but if we use the
    // function from there it will call the wrong reduce() function at the end.
    template <unsigned NLIMBS>
    static constexpr HOST_DEVICE_INLINE GoldilocksField from(const storage<NLIMBS>& xs)
    {
      static_assert(NLIMBS * 32 <= 576); // for now we support up to 576 bits
      storage<2 * TLC + 2> rs = {}; // we use 2*TLC+2 and not 2*TLC+1 because for now we don't support an odd number of
                                    // limbs in the storage struct
      int constexpr size = NLIMBS / TLC;
      // first reduction step:
      for (int i = 0; i < size; i++) // future optimization - because we assume a maximum value for size anyway, this
                                     // loop can be unrolled with potential performance benefits
      {
        const GoldilocksField& xi =
          *reinterpret_cast<const GoldilocksField*>(xs.limbs + i * TLC); // use casting instead of copying
        GoldilocksField pi =
          Base::get_reduced_digit_for_storage_reduction(i); // use precomputed values - pi = 2^(TLC*32*i) % p
        storage<2 * TLC + 2> temp = {};
        storage<2 * TLC>& temp_storage = *reinterpret_cast<storage<2 * TLC>*>(temp.limbs);
        icicle_math::template multiply_raw<TLC>(xi.limbs_storage, pi.limbs_storage, temp_storage); // multiplication
        icicle_math::template add_sub_limbs<2 * TLC + 2, false, false>(rs, temp, rs);              // accumulation
      }
      int constexpr extra_limbs = NLIMBS - TLC * size;
      if constexpr (extra_limbs > 0) { // handle the extra limbs (when TLC does not divide NLIMBS)
        const storage<extra_limbs>& xi = *reinterpret_cast<const storage<extra_limbs>*>(xs.limbs + size * TLC);
        GoldilocksField pi = Base::get_reduced_digit_for_storage_reduction(size);
        storage<2 * TLC + 2> temp = {};
        storage<extra_limbs + TLC>& temp_storage = *reinterpret_cast<storage<extra_limbs + TLC>*>(temp.limbs);
        icicle_math::template multiply_raw<extra_limbs, TLC>(xi, pi.limbs_storage, temp_storage); // multiplication
        icicle_math::template add_sub_limbs<2 * TLC + 2, false, false>(rs, temp, rs);             // accumulation
      }
      // second reduction step: - an alternative for this step would be to use the barret reduction straight away but
      // with a larger value of m.
      unsigned constexpr msbits_count = 2 * TLC * 32 - (2 * NBITS - 1);
      unsigned top_bits = (rs.limbs[2 * TLC] << msbits_count) + (rs.limbs[2 * TLC - 1] >> (32 - msbits_count));
      icicle_math::template add_sub_limbs<2 * TLC + 2, true, false>(
        rs, Base::get_mod_sub_for_storage_reduction(top_bits),
        rs); // subtracting the precomputed multiple of p from the look-up table
      // third and final step:
      storage<2 * TLC>& res = *reinterpret_cast<storage<2 * TLC>*>(rs.limbs);
      return typename Base::Wide{res}.reduce(); // finally, use goldilocks reduction
    }

    /* This is the non-template version of the from(storage) function above. It receives an array of bytes and its size
    and returns a field element after modular reduction. For now we support up to 576 bits. */
    static constexpr HOST_DEVICE_INLINE GoldilocksField from(const std::byte* in, unsigned nof_bytes)
    {
      storage<2 * TLC + 2> rs = {}; // we use 2*TLC+2 and not 2*TLC+1 because for now we don't support an odd number of
                                    // limbs in the storage struct
      unsigned constexpr bytes_per_field = TLC * 4;
      int size = nof_bytes / bytes_per_field;
      // first reduction step:
      for (int i = 0; i < size; i++) {
        const GoldilocksField& xi =
          *reinterpret_cast<const GoldilocksField*>(in + i * bytes_per_field); // use casting instead of copying
        GoldilocksField pi =
          Base::get_reduced_digit_for_storage_reduction(i); // use precomputed values - pi = 2^(TLC*32*i) % p
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
        GoldilocksField pi = Base::get_reduced_digit_for_storage_reduction(size);
        storage<2 * TLC + 2> temp = {};
        storage<2 * TLC>& temp_storage = *reinterpret_cast<storage<2 * TLC>*>(temp.limbs);
        icicle_math::template multiply_raw<TLC>(xi, pi.limbs_storage, temp_storage);  // multiplication
        icicle_math::template add_sub_limbs<2 * TLC + 2, false, false>(rs, temp, rs); // accumulation
      }
      // second reduction step: - an alternative for this step would be to use the barret reduction straight away but
      // with a larger value of m.
      unsigned constexpr msbits_count = 2 * TLC * 32 - (2 * NBITS - 1);
      unsigned top_bits = (rs.limbs[2 * TLC] << msbits_count) + (rs.limbs[2 * TLC - 1] >> (32 - msbits_count));
      icicle_math::template add_sub_limbs<2 * TLC + 2, true, false>(
        rs, Base::get_mod_sub_for_storage_reduction(top_bits),
        rs); // subtracting the precomputed multiple of p from the look-up table
      // third and final step:
      storage<2 * TLC>& res = *reinterpret_cast<storage<2 * TLC>*>(rs.limbs);
      return typename Base::Wide{res}.reduce(); // finally, use goldilocks reduction
    }

    // TODO: reinterpret cast
    HOST_DEVICE_INLINE GoldilocksField operator+(const GoldilocksField& ys) const
    {
      GoldilocksField rs = {};
      icicle_math::goldi_add(
        this->limbs_storage, ys.limbs_storage, Base::get_modulus(), Base::get_neg_modulus(), rs.limbs_storage);
      return rs;
    }

    HOST_DEVICE_INLINE GoldilocksField operator-(const GoldilocksField& ys) const
    {
      GoldilocksField rs = {};
      icicle_math::goldi_add(
        this->limbs_storage, ys.neg().limbs_storage, Base::get_modulus(), Base::get_neg_modulus(), rs.limbs_storage);
      return rs;
    }

    /*This function performs the goldilocks reduction:
    xs[63:0] + xs[95:64] * (2^32 - 1) - xs[127:96]
     */
    template <unsigned MODULUS_MULTIPLE = 1>
    constexpr HOST_DEVICE_INLINE GoldilocksField reduce() const
    {
      GoldilocksField rs = {};
      icicle_math::goldi_reduce(this->limbs_storage, Base::get_modulus(), Base::get_neg_modulus(), rs.limbs_storage);
      return static_cast<Base>(rs);
    }

    constexpr HOST_DEVICE_INLINE GoldilocksField inverse() const
    {
      if (*this == zero()) return zero();
      const GoldilocksField one = GoldilocksField{CONFIG::one};
      const GoldilocksField zero = GoldilocksField{CONFIG::zero};
      const ff_storage modulus = CONFIG::modulus;
      GoldilocksField u = *this;
      GoldilocksField v = GoldilocksField{modulus};
      GoldilocksField b = one;
      GoldilocksField c = {};
      while (!(u == one) && !(v == one)) {
        while (u.is_even()) {
          uint32_t carry = 0;
          u = u.div2();
          if (b.is_odd()) carry = Base::template add_limbs<TLC, true>(b.limbs_storage, modulus, b.limbs_storage);
          b = b.div2();
          if (carry) {
            b.limbs_storage.limbs[1] =
              b.limbs_storage.limbs[1] |
              (1U << 31); // If there is a carry then after the division by 2 we can insert it as the top bit
          }
        }
        while (v.is_even()) {
          uint32_t carry = 0;
          v = v.div2();
          if (c.is_odd()) carry = Base::template add_limbs<TLC, true>(c.limbs_storage, modulus, c.limbs_storage);
          c = c.div2();
          if (carry) {
            c.limbs_storage.limbs[1] =
              c.limbs_storage.limbs[1] |
              (1U << 31); // If there is a carry then after the division by 2 we can insert it as the top bit
          }
        }
        if (Base::lt(v, u)) {
          u = u - v;
          b = b - c;
        } else {
          v = v - u;
          c = c - b;
        }
      }
      return (u == one) ? b : c;
    }

    HOST_DEVICE_INLINE GoldilocksField operator*(const GoldilocksField& ys) const
    {
      return this->mul_wide(ys).reduce();
    }

    static HOST_INLINE GoldilocksField omega(uint32_t logn)
    {
      if (logn == 0) { return GoldilocksField{CONFIG::one}; }

      if (logn > CONFIG::omegas_count) {
        THROW_ICICLE_ERR(icicle::eIcicleError::INVALID_ARGUMENT, "ModArith: Invalid omega index");
      }

      GoldilocksField omega = GoldilocksField{CONFIG::rou};
      for (int i = 0; i < CONFIG::omegas_count - logn; i++) {
        omega = omega.sqr();
      }
      return omega;
    }

    static HOST_INLINE GoldilocksField omega_inv(uint32_t logn)
    {
      if (logn == 0) { return GoldilocksField{CONFIG::one}; }

      if (logn > CONFIG::omegas_count) {
        THROW_ICICLE_ERR(icicle::eIcicleError::INVALID_ARGUMENT, "ModArith: Invalid omega_inv index");
      }

      GoldilocksField omega = GoldilocksField{CONFIG::rou}.inverse();
      for (int i = 0; i < CONFIG::omegas_count - logn; i++)
        omega = omega.sqr();
      return omega;
    }

    static HOST_DEVICE_INLINE GoldilocksField inv_log_size(uint32_t logn) { return Base::inv_log_size(logn); }

    HOST_DEVICE_INLINE GoldilocksField sqr() const { return *this * *this; }

    HOST_DEVICE_INLINE GoldilocksField to_montgomery() const { return *this * GoldilocksField{CONFIG::montgomery_r}; }

    HOST_DEVICE_INLINE GoldilocksField from_montgomery() const
    {
      return *this * GoldilocksField{CONFIG::montgomery_r_inv};
    }

    constexpr HOST_DEVICE_INLINE GoldilocksField pow(int exp) const
    {
      GoldilocksField res = one();
      GoldilocksField base = *this;
      while (exp > 0) {
        if (exp & 1) res = res * base;
        base = base * base;
        exp >>= 1;
      }
      return res;
    }

    // Add a static reduce method to fix the Wide::reduce() call
    static constexpr HOST_DEVICE_INLINE GoldilocksField reduce(const typename Base::Wide& wide)
    {
      GoldilocksField rs = {};
      icicle_math::goldi_reduce(wide.limbs_storage, Base::get_modulus(), Base::get_neg_modulus(), rs.limbs_storage);
      return rs;
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
    // The next 2 parameters are unused and are only needed for compilation:
    static constexpr storage<limbs_count> m = {0x00000000, 0x00000000};
    static constexpr unsigned num_of_reductions = 0;
    static constexpr storage<2 * limbs_count> modulus_squared =
      params_gen::template get_square<limbs_count, 0>(modulus);
    static constexpr storage<2 * limbs_count> modulus_squared_2 =
      host_math::template left_shift<2 * limbs_count, 1>(modulus_squared);
    static constexpr storage<2 * limbs_count> modulus_squared_4 =
      host_math::template left_shift<2 * limbs_count, 1>(modulus_squared_2);
    static constexpr storage<limbs_count> modulus_2 = host_math::template left_shift<limbs_count, 1>(modulus);
    static constexpr storage<limbs_count> modulus_4 = host_math::template left_shift<limbs_count, 1>(modulus_2);
    MOD_SQR_SUBS()
    static constexpr storage_array<mod_subs_count, 2 * limbs_count + 2> mod_subs = {
      {{0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
       {0x7fffffff, 0x00000001, 0xffffffff, 0x7fffffff, 0x00000000, 0x00000000},
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

    // nonresidue to generate the extension field
    static constexpr uint32_t nonresidue = 7;
    // true if nonresidue is negative.
    static constexpr bool nonresidue_is_negative = false;
    static constexpr bool nonresidue_is_u32 = true;
  };

  /**
   * Scalar field. Is always a prime field.
   */
  typedef GoldilocksField<fp_config> scalar_t;

  template <class CONFIG, class T>
  class GoldilocksComplexExtensionField
  {
    friend T;

  public:
    typedef T FF;
    static constexpr unsigned TLC = 2 * FF::TLC;

    FF c0;
    FF c1;

    typedef typename FF::Wide FWide;

    struct Wide {
      FWide c0;
      FWide c1;

      static constexpr Wide HOST_DEVICE_INLINE from_field(const GoldilocksComplexExtensionField& xs)
      {
        return Wide{FWide::from_field(xs.c0), FWide::from_field(xs.c1)};
      }

      friend HOST_DEVICE_INLINE Wide operator+(const Wide& xs, const Wide& ys)
      {
        return Wide{xs.c0 + ys.c0, xs.c1 + ys.c1};
      }

      HOST_DEVICE_INLINE Wide operator-(const Wide& ys) const { return Wide{c0 - ys.c0, c1 - ys.c1}; }

      constexpr HOST_DEVICE_INLINE Wide neg() const { return Wide{FWide::neg(c0), FWide::neg(c1)}; }

      // Reduce the wide representation back to a GoldilocksComplexExtensionField element
      constexpr HOST_DEVICE_INLINE GoldilocksComplexExtensionField reduce() const
      {
        return GoldilocksComplexExtensionField{c0.reduce(), c1.reduce()};
      }
    };

    static constexpr HOST_DEVICE_INLINE GoldilocksComplexExtensionField zero()
    {
      return GoldilocksComplexExtensionField{FF::zero(), FF::zero()};
    }

    static constexpr HOST_DEVICE_INLINE GoldilocksComplexExtensionField one()
    {
      return GoldilocksComplexExtensionField{FF::one(), FF::zero()};
    }

    static constexpr HOST_DEVICE_INLINE GoldilocksComplexExtensionField from(uint32_t val)
    {
      return GoldilocksComplexExtensionField{FF::from(val), FF::zero()};
    }

    HOST_DEVICE_INLINE GoldilocksComplexExtensionField to_montgomery() const
    {
      return GoldilocksComplexExtensionField{c0.to_montgomery(), c1.to_montgomery()};
    }

    HOST_DEVICE_INLINE GoldilocksComplexExtensionField from_montgomery() const
    {
      return GoldilocksComplexExtensionField{c0.from_montgomery(), c1.from_montgomery()};
    }

    static HOST_INLINE GoldilocksComplexExtensionField rand_host()
    {
      return GoldilocksComplexExtensionField{FF::rand_host(), FF::rand_host()};
    }

    static void rand_host_many(GoldilocksComplexExtensionField* out, int size)
    {
      for (int i = 0; i < size; i++)
        out[i] = rand_host();
    }

    template <unsigned REDUCTION_SIZE = 1>
    static constexpr HOST_DEVICE_INLINE GoldilocksComplexExtensionField
    sub_modulus(const GoldilocksComplexExtensionField& xs)
    {
      return GoldilocksComplexExtensionField{
        FF::sub_modulus<REDUCTION_SIZE>(&xs.c0), FF::sub_modulus<REDUCTION_SIZE>(&xs.c1)};
    }

    friend std::ostream& operator<<(std::ostream& os, const GoldilocksComplexExtensionField& xs)
    {
      os << "{ Real: " << xs.c0 << " }; { Imaginary: " << xs.c1 << " }";
      return os;
    }

    friend HOST_DEVICE_INLINE GoldilocksComplexExtensionField
    operator+(GoldilocksComplexExtensionField xs, const GoldilocksComplexExtensionField& ys)
    {
      return GoldilocksComplexExtensionField{xs.c0 + ys.c0, xs.c1 + ys.c1};
    }

    friend HOST_DEVICE_INLINE GoldilocksComplexExtensionField
    operator-(GoldilocksComplexExtensionField xs, const GoldilocksComplexExtensionField& ys)
    {
      return GoldilocksComplexExtensionField{xs.c0 - ys.c0, xs.c1 - ys.c1};
    }

    friend HOST_DEVICE_INLINE GoldilocksComplexExtensionField
    operator+(FF xs, const GoldilocksComplexExtensionField& ys)
    {
      return GoldilocksComplexExtensionField{xs + ys.c0, ys.c1};
    }

    friend HOST_DEVICE_INLINE GoldilocksComplexExtensionField
    operator-(FF xs, const GoldilocksComplexExtensionField& ys)
    {
      return GoldilocksComplexExtensionField{xs - ys.c0, FF::neg(ys.c1)};
    }

    friend HOST_DEVICE_INLINE GoldilocksComplexExtensionField
    operator+(GoldilocksComplexExtensionField xs, const FF& ys)
    {
      return GoldilocksComplexExtensionField{xs.c0 + ys, xs.c1};
    }

    friend HOST_DEVICE_INLINE GoldilocksComplexExtensionField
    operator-(GoldilocksComplexExtensionField xs, const FF& ys)
    {
      return GoldilocksComplexExtensionField{xs.c0 - ys, xs.c1};
    }

    constexpr HOST_DEVICE_INLINE GoldilocksComplexExtensionField operator-() const
    {
      return GoldilocksComplexExtensionField{FF::neg(c0), FF::neg(c1)};
    }

    constexpr HOST_DEVICE_INLINE GoldilocksComplexExtensionField& operator+=(const GoldilocksComplexExtensionField& ys)
    {
      *this = *this + ys;
      return *this;
    }

    constexpr HOST_DEVICE_INLINE GoldilocksComplexExtensionField& operator-=(const GoldilocksComplexExtensionField& ys)
    {
      *this = *this - ys;
      return *this;
    }

    constexpr HOST_DEVICE_INLINE GoldilocksComplexExtensionField& operator*=(const GoldilocksComplexExtensionField& ys)
    {
      *this = *this * ys;
      return *this;
    }

    constexpr HOST_DEVICE_INLINE GoldilocksComplexExtensionField& operator+=(const FF& ys)
    {
      *this = *this + ys;
      return *this;
    }

    constexpr HOST_DEVICE_INLINE GoldilocksComplexExtensionField& operator-=(const FF& ys)
    {
      *this = *this - ys;
      return *this;
    }

    constexpr HOST_DEVICE_INLINE GoldilocksComplexExtensionField& operator*=(const FF& ys)
    {
      *this = *this * ys;
      return *this;
    }

    /*
     * Multiply by the quadratic non-residue used to construct the extension field.
     * For Goldilocks we currently have a small u32 non-residue (7), so we can use the
     * optimized unsigned multiplication variant that exists on the base field.
     */
    static constexpr HOST_DEVICE FF mul_by_nonresidue(const FF& xs)
    {
      if constexpr (CONFIG::nonresidue_is_u32) {
        return FF::template mul_unsigned<CONFIG::nonresidue>(xs);
      } else {
        return FF::template mul_const<CONFIG::nonresidue>(xs);
      }
    }

    static constexpr HOST_DEVICE FWide mul_by_nonresidue(const FWide& xs)
    {
      if constexpr (CONFIG::nonresidue_is_u32) {
        // First reduce to avoid potential overflow issues
        FF reduced = xs.reduce();
        // Then multiply by the nonresidue and convert back to wide
        return FF::template mul_unsigned<CONFIG::nonresidue>(reduced).mul_wide(FF::one());
      } else {
        FF reduced = xs.reduce();
        return reduced.mul_wide(FF::from(CONFIG::nonresidue));
      }
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    HOST_DEVICE_INLINE Wide mul_wide(const GoldilocksComplexExtensionField& ys) const
    {
      FWide real_prod = c0.mul_wide(ys.c0);
      FWide imaginary_prod = c1.mul_wide(ys.c1);
      FWide prod_of_sums = (c0 + c1).mul_wide(ys.c0 + ys.c1);
      FWide nonresidue_times_im = mul_by_nonresidue(imaginary_prod);
      return Wide{real_prod + nonresidue_times_im, prod_of_sums - real_prod - imaginary_prod};
    }

    // Non-templated version for FF type
    HOST_DEVICE_INLINE Wide mul_wide(const FF& ys) const { return Wide{c0.mul_wide(ys), c1.mul_wide(ys)}; }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE Wide mul_wide(const GoldilocksComplexExtensionField& xs, const FF& ys)
    {
      return Wide{FF::mul_wide(xs.c0, ys), FF::mul_wide(xs.c1, ys)};
    }

    template <unsigned MODULUS_MULTIPLE = 1>
    static constexpr HOST_DEVICE_INLINE Wide mul_wide(const FF& xs, const GoldilocksComplexExtensionField& ys)
    {
      return mul_wide(ys, xs);
    }

    HOST_DEVICE_INLINE GoldilocksComplexExtensionField reduce() const
    {
      return GoldilocksComplexExtensionField{c0.reduce(), c1.reduce()};
    }

    HOST_DEVICE_INLINE GoldilocksComplexExtensionField operator*(const GoldilocksComplexExtensionField& ys) const
    {
      Wide xy = mul_wide(ys);
      return xy.reduce();
    }

    HOST_DEVICE_INLINE GoldilocksComplexExtensionField operator*(const FF& ys) const
    {
      Wide xy = mul_wide(ys);
      return xy.reduce();
    }

    friend HOST_DEVICE_INLINE GoldilocksComplexExtensionField
    operator*(const FF& ys, const GoldilocksComplexExtensionField& xs)
    {
      return xs * ys;
    }

    friend HOST_DEVICE_INLINE bool
    operator==(const GoldilocksComplexExtensionField& xs, const GoldilocksComplexExtensionField& ys)
    {
      return (xs.c0 == ys.c0) && (xs.c1 == ys.c1);
    }

    friend HOST_DEVICE_INLINE bool
    operator!=(const GoldilocksComplexExtensionField& xs, const GoldilocksComplexExtensionField& ys)
    {
      return !(xs == ys);
    }

    HOST_DEVICE_INLINE GoldilocksComplexExtensionField sqr() const
    {
      // TODO: change to a more efficient squaring
      return *this * *this;
    }

    HOST_DEVICE_INLINE GoldilocksComplexExtensionField neg() const
    {
      return GoldilocksComplexExtensionField{c0.neg(), c1.neg()};
    }

    // inverse of zero is set to be zero which is what we want most of the time
    HOST_DEVICE_INLINE GoldilocksComplexExtensionField inverse() const
    {
      GoldilocksComplexExtensionField xs_conjugate = {c0, c1.neg()};
      FF nonresidue_times_im = mul_by_nonresidue(c1.sqr());
      nonresidue_times_im = CONFIG::nonresidue_is_negative ? nonresidue_times_im.neg() : nonresidue_times_im;
      // TODO: wide here
      FF xs_norm_squared = c0.sqr() - nonresidue_times_im;
      return xs_conjugate * GoldilocksComplexExtensionField{xs_norm_squared.inverse(), FF::zero()};
    }

    HOST_DEVICE_INLINE GoldilocksComplexExtensionField pow(int exp) const
    {
      GoldilocksComplexExtensionField res = GoldilocksComplexExtensionField::one();
      GoldilocksComplexExtensionField base = *this;
      while (exp > 0) {
        if (exp & 1) res = res * base;
        base = base * base;
        exp >>= 1;
      }
      return res;
    }

    template <unsigned NLIMBS>
    static constexpr HOST_DEVICE GoldilocksComplexExtensionField
    pow(GoldilocksComplexExtensionField base, storage<NLIMBS> exp)
    {
      GoldilocksComplexExtensionField res = one();
      while (host_math::is_zero(exp)) {
        if (host_math::get_bit<NLIMBS>(exp, 0)) res = res * base;
        base = base * base;
        exp = host_math::right_shift<NLIMBS, 1>(exp);
      }
      return res;
    }

    /* Receives an array of bytes and its size and returns extension field element. */
    static constexpr HOST_DEVICE_INLINE GoldilocksComplexExtensionField from(const std::byte* in, unsigned nof_bytes)
    {
      if (nof_bytes < 2 * sizeof(FF)) {
#ifndef __CUDACC__
        ICICLE_LOG_ERROR << "Input size is too small";
#endif // __CUDACC__
        return GoldilocksComplexExtensionField::zero();
      }
      return GoldilocksComplexExtensionField{FF::from(in, sizeof(FF)), FF::from(in + sizeof(FF), sizeof(FF))};
    }
  };

  /**
   * Complex extension field of `scalar_t` enabled if `-DEXT_FIELD` env variable is.
   */
  typedef GoldilocksComplexExtensionField<fp_config, scalar_t> extension_t;
} // namespace goldilocks

template <class CONFIG>
struct std::hash<goldilocks::GoldilocksField<CONFIG>> {
  std::size_t operator()(const goldilocks::GoldilocksField<CONFIG>& key) const
  {
    std::size_t hash = 0;
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

template <class CONFIG, class T>
struct SharedMemory<goldilocks::GoldilocksComplexExtensionField<CONFIG, T>> {
  __device__ goldilocks::GoldilocksComplexExtensionField<CONFIG, T>* getPointer()
  {
    extern __shared__ goldilocks::GoldilocksComplexExtensionField<CONFIG, T> s_ext2_scalar_[];
    return s_ext2_scalar_;
  }
};

#endif // __CUDACC__