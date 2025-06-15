#include "icicle/pairing/pairing.h"
#include "icicle/math/host_math.h"

#include "icicle/curves/curve_config.h"
using namespace curve_config;

#include "icicle/pairing/pairing_config.h"
using namespace pairing_config;

namespace icicle {
  template <>
  eIcicleError pairing<PairingConfig>(
    const PairingConfig::G1Affine& p, const PairingConfig::G2Affine& q, PairingConfig::TargetField& output)
  {
    auto coeffs = prepare_q<PairingConfig>(q);
    output = miller_loop<PairingConfig>(p, coeffs);
    final_exponentiation<PairingConfig>(output);
    return eIcicleError::SUCCESS;
  }

  extern "C" void
  CONCAT_EXPAND(ICICLE_FFI_PREFIX, pairing)(const affine_t* p, const g2_affine_t* q, PairingConfig::TargetField* output)
  {
    pairing<PairingConfig>(*p, *q, *output);
  }
} // namespace icicle