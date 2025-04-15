#pragma once

#include "icicle/math/host_math.h"

namespace host_math {

  /*This function implements the addition operation. The inputs are always in the range 0 to p except for when this
  function is called by the reduce function. In that case we can guarantee that one of the arguments is smaller than p
  so there is no overflow when adding (-p). The output is always between 0 an p.*/
  template <unsigned NLIMBS>
  static constexpr void goldi_add(
    const storage<NLIMBS>& xs,
    const storage<NLIMBS>& ys,
    const storage<NLIMBS>& mod,
    const storage<NLIMBS>& neg_mod,
    storage<NLIMBS>& rs)
  {
    auto carry = add_sub_limbs<NLIMBS, false, true>(xs, ys, rs); // Do the addition
    if (carry) {
      add_sub_limbs<NLIMBS, false, false>(rs, neg_mod, rs); // Adding (-p) effectively sutracts p in case there is a
                                                            // carry. This is guaranteed no to overflow.
    }
    if (__builtin_expect(
          rs.limbs64[0] >= mod.limbs64[0],
          0)) { // reducing into the range of 0 to p because icicle does not support the expanded representation for
                // now.
      rs.limbs64[0] = rs.limbs64[0] - mod.limbs64[0];
    }
  }

  /*This function performs the goldilocks reduction:
xs[63:0] + xs[95:64] * (2^32 - 1) - xs[127:96]
First it does the subtraction - xs[63:0] - xs[127:96] and hints the compiler that it is rare that xs[63:0] <
xs[127:96]. Then it adds xs[95:64] * (2^32 - 1) which is guaranteed to be smaller than p. This ensures that the
addition operation will not overflow. */
  template <unsigned NLIMBS>
  static constexpr void goldi_reduce(
    const storage<2 * NLIMBS>& xs, const storage<NLIMBS>& mod, const storage<NLIMBS>& neg_mod, storage<NLIMBS>& rs)
  {
    constexpr uint32_t gold_fact = uint32_t(-1); //(2^32 - 1)
    const storage<NLIMBS>& x_lo = *reinterpret_cast<const storage<NLIMBS>*>(xs.limbs);
    storage<NLIMBS> x_hi_hi = {xs.limbs[3]};
    auto carry = add_sub_limbs<NLIMBS, true, true>(x_lo, x_hi_hi, rs); // xs[63:0] - xs[127:96]
    if (__builtin_expect(carry, 0)) {
      add_sub_limbs<NLIMBS, true, false>(rs, neg_mod, rs); // cannot underflow
    }
    storage<NLIMBS> x_hi_lo = {};
    x_hi_lo.limbs64[0] =
      static_cast<uint64_t>(xs.limbs[2]) * static_cast<uint64_t>(gold_fact); // xs[95:64] * (2^32 - 1)

    storage<NLIMBS> rs2 = {};
    goldi_add(rs, x_hi_lo, mod, neg_mod, rs2);
    rs = rs2;
  }
} // namespace host_math