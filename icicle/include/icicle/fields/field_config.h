#pragma once

#include "icicle/fields/id.h"
#include "icicle/fields/field.h"

/**
 * @namespace field_config
 * Namespace with type definitions for finite fields. Here, concrete types are created in accordance
 * with the `-DFIELD` env variable passed during build.
 */
#if FIELD_ID == BN254
  #include "icicle/fields/snark_fields/bn254_scalar.h"
namespace field_config = bn254;
#elif FIELD_ID == BLS12_381
  #include "icicle/fields/snark_fields/bls12_381_scalar.h"
using bls12_381::fp_config;
namespace field_config = bls12_381;
#elif FIELD_ID == BLS12_377
  #include "icicle/fields/snark_fields/bls12_377_scalar.h"
namespace field_config = bls12_377;
#elif FIELD_ID == BW6_761
  #include "icicle/fields/snark_fields/bw6_761_scalar.h"
namespace field_config = bw6_761;
#elif FIELD_ID == GRUMPKIN
  #include "icicle/fields/snark_fields/grumpkin_scalar.h"
namespace field_config = grumpkin;

#elif FIELD_ID == BABY_BEAR
  #include "icicle/fields/stark_fields/babybear.h"
namespace field_config = babybear;
#elif FIELD_ID == STARK_252
  #include "icicle/fields/stark_fields/stark252.h"
namespace field_config = stark252;
#elif FIELD_ID == M31
  #include "icicle/fields/stark_fields/m31.h"
namespace field_config = m31;
#elif FIELD_ID == KOALA_BEAR
  #include "icicle/fields/stark_fields/koalabear.h"
namespace field_config = koalabear;
#elif FIELD_ID == GOLDILOCKS
  #include "icicle/fields/stark_fields/goldilocks.h"
namespace field_config = goldilocks;
#endif

// Note: rings are currently here since most code is shared with fields and include field_config.h
// Maybe we will refactor in the future

#if RING_ID == BABYKOALA
  #include "icicle/rings/params/babykoala.h"
namespace field_config = babykoala;
#endif
