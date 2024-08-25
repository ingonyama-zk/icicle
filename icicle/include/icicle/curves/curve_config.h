#pragma once

#include "icicle/fields/id.h"
#include "icicle/curves/projective.h"

/**
 * @namespace curve_config
 * Namespace with type definitions for short Weierstrass pairing-friendly [elliptic
 * curves](https://hyperelliptic.org/EFD/g1p/auto-shortw.html). Here, concrete types are created in accordance
 * with the `-DCURVE` env variable passed during build.
 */
#if CURVE_ID == BN254
#include "icicle/curves/params/bn254.h"
namespace curve_config = bn254;

#elif CURVE_ID == BLS12_381
#include "icicle/curves/params/bls12_381.h"
namespace curve_config = bls12_381;

#elif CURVE_ID == BLS12_377
#include "icicle/curves/params/bls12_377.h"
namespace curve_config = bls12_377;

#elif CURVE_ID == BW6_761
#include "icicle/curves/params/bw6_761.h"
namespace curve_config = bw6_761;

#elif CURVE_ID == GRUMPKIN
#include "icicle/curves/params/grumpkin.h"
namespace curve_config = grumpkin;
#endif