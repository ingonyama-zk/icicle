#include "icicle/pairings/pairing.h"
#include "icicle/math/host_math.h"

#include "icicle/curves/curve_config.h"
using namespace curve_config;

#include "icicle/pairings/pairing_config.h"
using namespace pairing_config;

namespace icicle {
  // template <
  //   typename Pairing,
  //   typename TF = typename Pairing::target_field_t,
  //   typename TargetAffine = typename Pairing::target_affine_t>
  // TF miller_loop(const TargetAffine& p, const TargetAffine& q)
  // {
  //   TF f = TF::one();

  //   if (TargetAffine::zero() == p || TargetAffine::zero() == q) { return f; }

  //   TF two = TF::from(2);
  //   TF three = TF::from(3);

  //   TF xt = p.x;
  //   TF yt = p.y;
  //   TF x2t, y2t, xtp, ytp;

  //   for (int j = Pairing::R_BITS; j > 0; j--) {
  //     TF m = TF::sqr(xt) * three * TF::inverse(two * yt);
  //     x2t = TF::sqr(m) - two * xt;
  //     y2t = TF::neg(yt) - m * (x2t - xt);
  //     f = TF::sqr(f) * (q.y - yt - m * (q.x - xt));
  //     xt = x2t;
  //     yt = y2t;

  //     if (host_math::get_bit(Pairing::R, j - 1)) {
  //       m = (yt - p.y) * TF::inverse(xt - p.x);
  //       xtp = TF::sqr(m) - xt - p.x;
  //       ytp = TF::neg(yt) - m * (xtp - xt);
  //       f = f * (q.y - yt - m * (q.x - xt));
  //       xt = xtp;
  //       yt = ytp;
  //     }
  //   }
  //   f = f * (q.x - xt);
  //   return f;
  // }

  template <>
  eIcicleError pairing<affine_t, g2_affine_t, PairingConfig>(
    const affine_t& p, const g2_affine_t& q, PairingConfig::TargetField* output)
  {
    auto coeffs = prepare_q<PairingConfig>(q);
    typename PairingConfig::TargetField f = miller_loop<PairingConfig>(p, coeffs);
    final_exponentiation<PairingConfig>(f);
    *output = f;
    return eIcicleError::SUCCESS;
  }
} // namespace icicle