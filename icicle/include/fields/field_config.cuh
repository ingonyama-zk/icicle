#pragma once
#ifndef FIELD_CONFIG_H
#define FIELD_CONFIG_H

#include "fields/id.h"

#include "fields/field.cuh"
#if defined(EXT_FIELD)
#include "fields/extension_field.cuh"
#endif

#if FIELD_ID == BN254
#include "fields/snark_fields/bn254_scalar.cuh"
using bn254::fp_config;
#elif FIELD_ID == BLS12_381
#include "fields/snark_fields/bls12_381_scalar.cuh"
using bls12_381::fp_config;
#elif FIELD_ID == BLS12_377
#include "fields/snark_fields/bls12_377_scalar.cuh"
using bls12_377::fp_config;
#elif FIELD_ID == BW6_761
#include "fields/snark_fields/bls12_377_base.cuh"
typedef bls12_377::fq_config fp_config;
#elif FIELD_ID == GRUMPKIN
#include "fields/snark_fields/bn254_base.cuh"
typedef bn254::fq_config fp_config;
#elif FIELD_ID == BABY_BEAR
#include "fields/stark_fields/baby_bear.cuh"
using baby_bear::fp_config;
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