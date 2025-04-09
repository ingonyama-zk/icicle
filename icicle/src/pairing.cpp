#include "icicle/pairings/pairing.h"
#include "icicle/math/host_math.h"

#include "icicle/curves/curve_config.h"
using namespace curve_config;

#include "icicle/pairings/pairing_config.h"
using namespace pairing_config;

namespace icicle {
  template <>
  eIcicleError pairing<PairingConfig>(
    const PairingConfig::G1Affine& p, const PairingConfig::G2Affine& q, PairingConfig::TargetField* output)
  {
    auto coeffs = prepare_q<PairingConfig>(q);
    typename PairingConfig::TargetField f = miller_loop<PairingConfig>(p, coeffs);
    final_exponentiation<PairingConfig>(f);
    *output = f;
    return eIcicleError::SUCCESS;
  }
} // namespace icicle