#pragma once

#include "icicle/fields/storage.h"
#include "icicle/fields/host_math.h"

namespace params_gen {
  template <unsigned NLIMBS, unsigned BIT_SHIFT>
  static constexpr HOST_INLINE storage<2 * NLIMBS> get_square(const storage<NLIMBS>& xs)
  {
    storage<2 * NLIMBS> rs = {};
    host_math::template multiply_raw<NLIMBS, NLIMBS, true>(xs, xs, rs);
    return host_math::template left_shift<2 * NLIMBS, BIT_SHIFT>(rs);
  }

  template <unsigned NLIMBS>
  static constexpr HOST_INLINE storage<NLIMBS>
  get_difference_no_carry(const storage<NLIMBS>& xs, const storage<NLIMBS>& ys)
  {
    storage<NLIMBS> rs = {};
    host_math::template add_sub_limbs<NLIMBS, true, false, true>(xs, ys, rs);
    return rs;
  }

  template <unsigned NLIMBS, unsigned EXP>
  static constexpr HOST_INLINE storage<NLIMBS> get_m(const storage<NLIMBS>& modulus)
  {
    storage<NLIMBS> rs = {};
    storage<NLIMBS> qs = {};
    storage<2 * NLIMBS> wide_one = {1};
    storage<2 * NLIMBS> pow_of_2 = host_math::template left_shift<2 * NLIMBS, EXP>(wide_one);
    host_math::template integer_division<2 * NLIMBS, NLIMBS, NLIMBS, true>(pow_of_2, modulus, qs, rs);
    return qs;
  }

  template <unsigned NLIMBS, bool INV>
  static constexpr HOST_INLINE storage<NLIMBS> get_montgomery_constant(const storage<NLIMBS>& modulus)
  {
    storage<NLIMBS> rs = {1};
    for (int i = 0; i < 32 * NLIMBS; i++) {
      if (INV) {
        if (rs.limbs[0] & 1) host_math::template add_sub_limbs<NLIMBS, false, false, true>(rs, modulus, rs);
        rs = host_math::template right_shift<NLIMBS, 1>(rs);
      } else {
        rs = host_math::template left_shift<NLIMBS, 1>(rs);
        storage<NLIMBS> temp = {};
        rs = host_math::template add_sub_limbs<NLIMBS, true, true, true>(rs, modulus, temp) ? rs : temp;
      }
    }
    return rs;
  }

  constexpr unsigned floorlog2(uint32_t x) { return x == 1 ? 0 : 1 + floorlog2(x >> 1); }

  template <unsigned NLIMBS, unsigned NBITS>
  constexpr unsigned num_of_reductions(const storage<NLIMBS>& modulus, const storage<NLIMBS>& m)
  {
    storage<2 * NLIMBS> x1 = {};
    storage<3 * NLIMBS> x2 = {};
    storage<3 * NLIMBS> x3 = {};
    host_math::template multiply_raw<NLIMBS, NLIMBS, true>(modulus, m, x1);
    host_math::template multiply_raw<NLIMBS, 2 * NLIMBS, true>(modulus, x1, x2);
    storage<2 * NLIMBS> one = {1};
    storage<2 * NLIMBS> pow_of_2 = host_math::template left_shift<2 * NLIMBS, NBITS>(one);
    host_math::template multiply_raw<NLIMBS, 2 * NLIMBS, true>(modulus, pow_of_2, x3);
    host_math::template add_sub_limbs<3 * NLIMBS, true, false, true>(x3, x2, x2);
    double err = (double)x2.limbs[2 * NLIMBS - 1] / pow_of_2.limbs[2 * NLIMBS - 1];
    err += (double)m.limbs[NLIMBS - 1] / 0xffffffff;
    err += (double)NLIMBS / 0x80000000;
    return unsigned(err) + 1;
  }

  template <unsigned NLIMBS>
  constexpr unsigned two_adicity(const storage<NLIMBS>& modulus)
  {
    unsigned two_adicity = 1;
    storage<NLIMBS> temp = host_math::template right_shift<NLIMBS, 1>(modulus);
    while (!(temp.limbs[0] & 1)) {
      temp = host_math::template right_shift<NLIMBS, 1>(temp);
      two_adicity++;
    }
    return two_adicity;
  }

  template <unsigned NLIMBS, unsigned TWO_ADICITY>
  constexpr storage_array<TWO_ADICITY, NLIMBS> get_invs(const storage<NLIMBS>& modulus)
  {
    storage_array<TWO_ADICITY, NLIMBS> invs = {};
    storage<NLIMBS> rs = {1};
    for (int i = 0; i < TWO_ADICITY; i++) {
      if (rs.limbs[0] & 1) host_math::template add_sub_limbs<NLIMBS, false, false, true>(rs, modulus, rs);
      rs = host_math::template right_shift<NLIMBS, 1>(rs);
      invs.storages[i] = rs;
    }
    return invs;
  }

  // template <unsigned NLIMBS, unsigned reduced_digits_count, unsigned mod_sqr_bit_count>
  // constexpr storage_array<reduced_digits_count, NLIMBS+2> get_modulus_sqr_subs(const storage<NLIMBS>& modulus_sqr)
  // {
  //   storage_array<reduced_digits_count, NLIMBS+2> mod_subs = {};
  //   storage<NLIMBS+2> curr = {0};
  //   storage<NLIMBS+2> mod_sqr_inc = {0};
  //   storage<NLIMBS+2> next_mod_sqr_inc = {0};
  //   storage<NLIMBS+2> mod_sqr_extended = {0};
  //   for (int i = 0; i < NLIMBS; i++)
  //   {
  //     mod_sqr_extended.limbs[i] = modulus_sqr.limbs[i];
  //   }
  //   mod_subs.storages[0] = {0};
  //   for (int i = 1; i < reduced_digits_count; i++) {
  //     host_math::template add_sub_limbs<NLIMBS+2, false, false, true>(mod_sqr_inc, mod_sqr_extended, next_mod_sqr_inc);
  //     storage<NLIMBS+2> next = host_math::template right_shift<NLIMBS+2, mod_sqr_bit_count>(next_mod_sqr_inc);
  //     if (curr.limbs[0] != next.limbs[0]) { //only lsb limb is non-zero
  //       mod_subs.storages[i] = curr;
  //       curr = next;
  //     }
  //     mod_sqr_inc = next_mod_sqr_inc;
  //   }
  //   return mod_subs;
  // }

  template <unsigned NLIMBS, unsigned mod_subs_count, unsigned mod_bit_count>
  constexpr storage_array<mod_subs_count, 2*NLIMBS+2> get_modulus_subs(const storage<NLIMBS>& modulus)
  {
    storage_array<mod_subs_count, 2*NLIMBS+2> mod_subs = {};
    unsigned constexpr bit_shift = 2*mod_bit_count-1;
    mod_subs.storages[0] = {0};
    for (int i = 1; i < mod_subs_count; i++) {
      storage<2*NLIMBS+2> temp = {};
      storage<NLIMBS> rs = {};
      storage<NLIMBS+2> mod_sub_factor = {};
      temp.limbs[0] = i;
      storage<2*NLIMBS+2> candidate = host_math::template left_shift<2*NLIMBS+2, bit_shift>(temp);
      host_math::template integer_division<2*NLIMBS+2, NLIMBS, NLIMBS+2, true>(candidate, modulus, mod_sub_factor, rs);
      storage<2*NLIMBS+2> temp2 = {};
      host_math::template multiply_raw<NLIMBS+2, NLIMBS, true>(mod_sub_factor, modulus, temp2);
      mod_subs.storages[i] = temp2;
    }
    return mod_subs;
  }
} // namespace params_gen

#define PARAMS(modulus)                                                                                                \
  static constexpr unsigned limbs_count = modulus.LC;                                                                  \
  static constexpr unsigned modulus_bit_count =                                                                        \
    32 * (limbs_count - 1) + params_gen::floorlog2(modulus.limbs[limbs_count - 1]) + 1;                                \
  static constexpr storage<limbs_count> zero = {};                                                                     \
  static constexpr storage<limbs_count> one = {1};                                                                     \
  static constexpr storage<limbs_count> modulus_2 = host_math::template left_shift<limbs_count, 1>(modulus);           \
  static constexpr storage<limbs_count> modulus_4 = host_math::template left_shift<limbs_count, 1>(modulus_2);         \
  static constexpr storage<limbs_count> neg_modulus =                                                                  \
    params_gen::template get_difference_no_carry<limbs_count>(zero, modulus);                                          \
  static constexpr storage<2 * limbs_count> modulus_squared =                                                          \
    params_gen::template get_square<limbs_count, 0>(modulus);                                                          \
  static constexpr storage<2 * limbs_count> modulus_squared_2 =                                                        \
    host_math::template left_shift<2 * limbs_count, 1>(modulus_squared);                                               \
  static constexpr storage<2 * limbs_count> modulus_squared_4 =                                                        \
    host_math::template left_shift<2 * limbs_count, 1>(modulus_squared_2);                                             \
  static constexpr storage<limbs_count> m = params_gen::template get_m<limbs_count, 2 * modulus_bit_count>(modulus);   \
  static constexpr storage<limbs_count> montgomery_r =                                                                 \
    params_gen::template get_montgomery_constant<limbs_count, false>(modulus);                                         \
  static constexpr storage<limbs_count> montgomery_r_inv =                                                             \
    params_gen::template get_montgomery_constant<limbs_count, true>(modulus);                                          \
  static constexpr unsigned num_of_reductions =                                                                        \
    params_gen::template num_of_reductions<limbs_count, 2 * modulus_bit_count>(modulus, m);                            

#define MOD_SQR_SUBS()    \
  static constexpr unsigned mod_subs_count = reduced_digits_count<<(limbs_count*32+1-modulus_bit_count); 
  // static constexpr storage_array<mod_subs_count, 2 * limbs_count + 2> mod_subs =                                 \
  //   params_gen::template get_modulus_subs<limbs_count, mod_subs_count, modulus_bit_count>(modulus);

#define TWIDDLES(modulus, rou)                                                                                         \
  static constexpr unsigned omegas_count = params_gen::template two_adicity<limbs_count>(modulus);                     \
  static constexpr storage_array<omegas_count, limbs_count> inv =                                                      \
    params_gen::template get_invs<limbs_count, omegas_count>(modulus);
