#pragma once

#include "icicle/fields/id.h"

#if FIELD_ID == BN254
  #include "icicle/fields/snark_fields/bn254_tower.h"
namespace tower_config = bn254;
#elif FIELD_ID == BLS12_381
  #include "icicle/fields/snark_fields/bls12_381_tower.h"
using bls12_381::fp_config;
namespace tower_config = bls12_381;
#elif FIELD_ID == BLS12_377
  #include "icicle/fields/snark_fields/bls12_377_tower.h"
namespace tower_config = bls12_377;
#endif

/**
 * @namespace pairing_config
 * Namespace with pairing implementations. Here, concrete algorithms are chosen in accordance
 * with the `-DCURVE` env variable passed during build.
 */
#if CURVE_ID == BN254
  #include "icicle/pairing/params/bn254.h"
namespace pairing_config = pairing_bn254;

#elif CURVE_ID == BLS12_381
  #include "icicle/pairing/params/bls12_381.h"
namespace pairing_config = pairing_bls12_381;

#elif CURVE_ID == BLS12_377
  #include "icicle/pairing/params/bls12_377.h"
namespace pairing_config = pairing_bls12_377;
#endif