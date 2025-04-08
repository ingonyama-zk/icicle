#pragma once

#include "icicle/fields/id.h"

/**
 * @namespace pairing_config
 * Namespace with pairing implementations. Here, concrete algorithms are chosen in accordance
 * with the `-DCURVE` env variable passed during build.
 */
#if CURVE_ID == BN254
  #include "icicle/pairings/params/bn254.h"
namespace pairing_config = pairing_bn254;

#elif CURVE_ID == BLS12_381
  #include "icicle/pairings/params/bls12_381.h"
namespace pairing_config = pairing_bls12_381;

#elif CURVE_ID == BLS12_377
  #include "icicle/pairings/params/bls12_377.h"
namespace pairing_config = pairing_bls12_377;

#elif CURVE_ID == BW6_761
  #include "icicle/pairings/params/bw6_761.h"
namespace pairing_config = pairing_bw6_761;
#endif