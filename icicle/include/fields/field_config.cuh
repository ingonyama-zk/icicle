#pragma once
#ifndef FIELD_CONFIG_H
#define FIELD_CONFIG_H

#include "fields/id.h"

#include "fields/field.cuh"
#if defined(EXT_FIELD)
#include "fields/extension_field.cuh"
#endif

#if FIELD_ID == BN254
#include "fields/snark_fields/bn254_fields.cuh"
using namespace bn254;
#elif FIELD_ID == BLS12_381
#include "fields/snark_fields/bls12_381_fields.cuh"
using namespace bls12_381;
#elif FIELD_ID == BLS12_377
#include "fields/snark_fields/bls12_377_fields.cuh"
using namespace bls12_377;
#elif FIELD_ID == BW6_761
#include "fields/snark_fields/bw6_761_fields.cuh"
using namespace bw6_761;
#elif FIELD_ID == GRUMPKIN
#include "fields/snark_fields/grumpkin_fields.cuh"
using namespace grumpkin;
#elif FIELD_ID == BABY_BEAR
#include "fields/stark_fields/baby_bear.cuh"
using namespace baby_bear;
#endif

/**
 * @namespace field_config
 * Namespace with type definitions for finite fields. Here, concrete types are created in accordance
 * with the `-DFIELD` env variable passed during build.
 */
namespace field_config {
  /**
   * Scalar field. Is always a prime field.
   */
  typedef Field<fp_config> scalar_t;
} // namespace field_config

#endif