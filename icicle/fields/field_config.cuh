#pragma once
#ifndef INDEX_H
#define INDEX_H

#define BN254     1
#define BLS12_381 2
#define BLS12_377 3
#define BW6_761   4
#define GRUMPKIN  5

#include "field.cuh"
#if defined(EXT_DEFINED)
#include "extension_field.cuh"
#endif

#if FIELD_ID == BN254
#include "bn254_params.cuh"
using namespace bn254;
#elif FIELD_ID == BLS12_381
#include "bls12_381_params.cuh"
using namespace bls12_381;
#elif FIELD_ID == BLS12_377
#include "bls12_377_params.cuh"
using namespace bls12_377;
#elif FIELD_ID == BW6_761
#include "bw6_761_params.cuh"
using namespace bw6_761;
#elif FIELD_ID == GRUMPKIN
#include "grumpkin_params.cuh"
using namespace grumpkin;
#endif

/**
 * @namespace curve_config
 * Namespace with type definitions for short Weierstrass pairing-friendly [elliptic
 * curves](https://hyperelliptic.org/EFD/g1p/auto-shortw.html). Here, concrete types are created in accordance
 * with the `-DFIELD` env variable passed during build.
 */
namespace field_config {
  /**
   * Scalar field of the curve. Is always a prime field.
   */
  typedef Field<fp_config> scalar_t;

  #ifdef CURVE_FIELDS
  /**
   * Base field of G1 curve. Is always a prime field.
   */
  typedef Field<fq_config> point_field_t;
  #endif

#if defined(EXT_DEFINED)
#if FIELD_ID == BW6_761
  typedef point_field_t g2_point_field_t;
#else
  typedef ExtensionField<fq_config> g2_point_field_t;
#endif
#endif

} // namespace curve_config

#endif