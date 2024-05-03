#pragma once
#ifndef PARAMS_GEN_H
#define PARAMS_GEN_H

#include "storage.cuh"
#include "host_math.cuh"

using namespace host_math;

namespace params_gen {
  template <unsigned NLIMBS, unsigned BIT_SHIFT>
  static constexpr HOST_INLINE storage<2 * NLIMBS> get_square(const storage<NLIMBS>& xs)
  {
    storage<2 * NLIMBS> rs = {};
    multiply_raw_host<NLIMBS>(xs, xs, rs);
    return left_shift<2 * NLIMBS, BIT_SHIFT>(rs);
  }

  template <unsigned NLIMBS>
  static constexpr HOST_INLINE storage<NLIMBS> get_difference_no_carry(const storage<NLIMBS>& xs, const storage<NLIMBS>& ys)
  {
    storage<NLIMBS> rs = {};
    add_sub_limbs_host<NLIMBS, true, false>(xs, ys, rs);
    return rs;
  }

  template <unsigned NLIMBS, unsigned EXP>
  static constexpr HOST_INLINE storage<NLIMBS> get_m(const storage<NLIMBS>& modulus)
  {
    storage<NLIMBS> rs = {};
    storage<NLIMBS> qs = {};
    storage<2 * NLIMBS> wide_one = {1};
    storage<2 * NLIMBS> pow_of_2 = left_shift<2 * NLIMBS, EXP>(wide_one);
    integer_division_host<2 * NLIMBS, NLIMBS>(pow_of_2, modulus, qs, rs);
    return qs;
  }

  template <unsigned NLIMBS, bool INV>
  static constexpr HOST_INLINE storage<NLIMBS> get_montgomery_constant(const storage<NLIMBS>& modulus)
  {
    storage<NLIMBS> rs = {1};
    for (int i = 0; i < 32 * NLIMBS; i++) {
      if (INV) {
        if (rs.limbs[0] & 1)
          add_sub_limbs_host<NLIMBS, false, false>(rs, modulus, rs);
        rs = right_shift<NLIMBS, 1>(rs);
      } else {
        rs = left_shift<NLIMBS, 1>(rs);
        storage<NLIMBS> temp = {};
        rs = add_sub_limbs_host<NLIMBS, true, true>(rs, modulus, temp) ? rs : temp;
      }
    }
    return rs;
  }

  constexpr unsigned floorlog2(uint32_t x)
  {
    return x == 1 ? 0 : 1 + floorlog2(x >> 1);
  }

  template <unsigned NLIMBS, unsigned NBITS>
  constexpr unsigned num_of_reductions(const storage<NLIMBS>& modulus, const storage<NLIMBS>& m)
  {
    storage<2 * NLIMBS> x1 = {};
    storage<3 * NLIMBS> x2 = {};
    storage<3 * NLIMBS> x3 = {};
    multiply_raw_host<NLIMBS>(modulus, m, x1);
    multiply_raw_host<NLIMBS, 2 * NLIMBS>(modulus, x1, x2);
    storage<2 * NLIMBS> one = {1};
    storage<2 * NLIMBS> pow_of_2 = left_shift<2 * NLIMBS, NBITS>(one);
    multiply_raw_host<NLIMBS, 2 * NLIMBS>(modulus, pow_of_2, x3);
    add_sub_limbs_host<3 * NLIMBS, true, false>(x3, x2, x2);
    // static_assert(x2.limbs[2 * NLIMBS] == 0, "error in estimating error");
    double err = (double)x2.limbs[2 * NLIMBS - 1] / pow_of_2.limbs[2 * NLIMBS - 1];
    err += (double)m.limbs[NLIMBS - 1] / 0xffffffff;
    err += (double)NLIMBS / 0x80000000;
    return unsigned(err) + 1;
  }

  template <unsigned NLIMBS>
  constexpr unsigned two_adicity(const storage<NLIMBS>& modulus)
  {
    unsigned two_adicity = 1;
    storage<NLIMBS> temp = right_shift<NLIMBS, 1>(modulus);
    while (!(temp.limbs[0] & 1)) {
      temp = right_shift<NLIMBS, 1>(temp);
      two_adicity++;
    }
    return two_adicity;
  }

  template <unsigned NLIMBS, unsigned TWO_ADICITY>
  constexpr storage_array<TWO_ADICITY, NLIMBS> get_invs(const storage<NLIMBS>& modulus) {
    storage_array<TWO_ADICITY, NLIMBS> invs = {};
    storage<NLIMBS> rs = {1};
    for (int i = 0; i < TWO_ADICITY; i++) {
      if (rs.limbs[0] & 1)
        add_sub_limbs_host<NLIMBS, false, false>(rs, modulus, rs);
      rs = right_shift<NLIMBS, 1>(rs);
      invs.storages[i] = rs;
    }
    return invs;
  }
} // namespace params_gen

#define PARAMS(modulus) static constexpr unsigned limbs_count = modulus.LC; \
                        static constexpr unsigned modulus_bit_count = 32 * (limbs_count - 1) + params_gen::floorlog2(modulus.limbs[limbs_count - 1]) + 1; \
                        static constexpr storage<limbs_count> zero = {}; \
                        static constexpr storage<limbs_count> one = {1}; \
                        static constexpr storage<limbs_count> modulus_2 = host_math::template left_shift<limbs_count, 1>(modulus); \
                        static constexpr storage<limbs_count> modulus_4 = host_math::template left_shift<limbs_count, 1>(modulus_2); \
                        static constexpr storage<limbs_count> neg_modulus = params_gen::template get_difference_no_carry<limbs_count>(zero, modulus); \
                        static constexpr storage<2 * limbs_count> modulus_squared = params_gen::template get_square<limbs_count, 0>(modulus); \
                        static constexpr storage<2 * limbs_count> modulus_squared_2 = host_math::template left_shift<2 * limbs_count, 1>(modulus_squared); \
                        static constexpr storage<2 * limbs_count> modulus_squared_4 = host_math::template left_shift<2 * limbs_count, 1>(modulus_squared_2); \
                        static constexpr storage<2 * limbs_count> modulus_squared_8 = host_math::template left_shift<2 * limbs_count, 1>(modulus_squared_4); \
                        static constexpr storage<2 * limbs_count> modulus_squared_16 = host_math::template left_shift<2 * limbs_count, 1>(modulus_squared_8); \
                        static constexpr storage<limbs_count> m = params_gen::template get_m<limbs_count, 2 * modulus_bit_count>(modulus); \
                        static constexpr storage<limbs_count> m_2 = params_gen::template get_m<limbs_count, 2 * modulus_bit_count + 1>(modulus); \
                        static constexpr storage<limbs_count> m_4 = params_gen::template get_m<limbs_count, 2 * modulus_bit_count + 2>(modulus); \
                        static constexpr storage<limbs_count> m_8 = params_gen::template get_m<limbs_count, 2 * modulus_bit_count + 3>(modulus); \
                        static constexpr storage<limbs_count> m_16 = params_gen::template get_m<limbs_count, 2 * modulus_bit_count + 4>(modulus); \
                        static constexpr storage<limbs_count> m_32 = params_gen::template get_m<limbs_count, 2 * modulus_bit_count + 5>(modulus); \
                        static constexpr storage<limbs_count> montgomery_r = params_gen::template get_montgomery_constant<limbs_count, false>(modulus); \
                        static constexpr storage<limbs_count> montgomery_r_inv = params_gen::template get_montgomery_constant<limbs_count, true>(modulus); \
                        static constexpr unsigned num_of_reductions = params_gen::template num_of_reductions<limbs_count, 2 * modulus_bit_count>(modulus, m);

#define TWIDDLES(modulus, rou) static constexpr unsigned omegas_count = params_gen::template two_adicity<limbs_count>(modulus); \
                               static constexpr storage_array<omegas_count, limbs_count> inv = params_gen::template get_invs<limbs_count, omegas_count>(modulus);

#endif