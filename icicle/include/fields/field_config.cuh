#pragma once
#ifndef FIELD_CONFIG_H
#define FIELD_CONFIG_H

#include "fields/id.h"
#include "gpu-utils/sharedmem.cuh"

#include "fields/field.cuh"
#if defined(EXT_FIELD)
#include "fields/extension_field.cuh"
#endif

#if FIELD_ID == BN254_FIELDS
#include "fields/params/bn254_fields.cuh"
using namespace bn254;
#elif FIELD_ID == BLS12_381_FIELDS
#include "fields/params/bls12_381_fields.cuh"
using namespace bls12_381;
#elif FIELD_ID == BLS12_377_FIELDS
#include "fields/params/bls12_377_fields.cuh"
using namespace bls12_377;
#elif FIELD_ID == BW6_761_FIELDS
#include "fields/params/bw6_761_fields.cuh"
using namespace bw6_761;
#elif FIELD_ID == GRUMPKIN_FIELDS
#include "fields/params/grumpkin_fields.cuh"
using namespace grumpkin;
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