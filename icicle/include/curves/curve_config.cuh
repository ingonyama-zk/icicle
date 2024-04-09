#pragma once
#ifndef CURVE_CONFIG_H
#define CURVE_CONFIG_H

#include "fields/id.h"
#include "curves/projective.cuh"

#if CURVE_ID == BN254
#include "curves/params/bn254.cuh"
using namespace bn254;

#elif CURVE_ID == BLS12_381
#include "curves/params/bls12_381.cuh"
using namespace bls12_381;

#elif CURVE_ID == BLS12_377
#include "curves/params/bls12_377.cuh"
using namespace bls12_377;

#elif CURVE_ID == BW6_761
#include "curves/params/bw6_761.cuh"
using namespace bw6_761;

#elif CURVE_ID == GRUMPKIN
#include "curves/params/grumpkin.cuh"
using namespace grumpkin;
#endif

#include "fields/field_config.cuh"
using field_config::scalar_t;

/**
 * @namespace curve_config
 * Namespace with type definitions for short Weierstrass pairing-friendly [elliptic
 * curves](https://hyperelliptic.org/EFD/g1p/auto-shortw.html). Here, concrete types are created in accordance
 * with the `-DCURVE` env variable passed during build.
 */
namespace curve_config {
  /**
   * Base field of G1 curve. Is always a prime field.
   */
  typedef Field<fq_config> point_field_t;

  static constexpr point_field_t generator_x = point_field_t{g1_gen_x};
  static constexpr point_field_t generator_y = point_field_t{g1_gen_y};
  static constexpr point_field_t b = point_field_t{weierstrass_b};
  /**
   * [Projective representation](https://hyperelliptic.org/EFD/g1p/auto-shortw-projective.html)
   * of G1 curve consisting of three coordinates of type [point_field_t](point_field_t).
   */
  typedef Projective<point_field_t, scalar_t, b, generator_x, generator_y> projective_t;
  /**
   * Affine representation of G1 curve consisting of two coordinates of type [point_field_t](point_field_t).
   */
  typedef Affine<point_field_t> affine_t;

#ifdef G2
#if CURVE_ID == BW6_761
  typedef point_field_t g2_point_field_t;
  static constexpr g2_point_field_t g2_generator_x = g2_point_field_t{g2_gen_x};
  static constexpr g2_point_field_t g2_generator_y = g2_point_field_t{g2_gen_y};
  static constexpr g2_point_field_t g2_b = g2_point_field_t{g2_weierstrass_b};
#else
#include "fields/quadratic_extension.cuh"

  typedef ExtensionField<fq_config> g2_point_field_t;
  static constexpr g2_point_field_t g2_generator_x =
    g2_point_field_t{point_field_t{g2_gen_x_re}, point_field_t{g2_gen_x_im}};
  static constexpr g2_point_field_t g2_generator_y =
    g2_point_field_t{point_field_t{g2_gen_y_re}, point_field_t{g2_gen_y_im}};
  static constexpr g2_point_field_t g2_b =
    g2_point_field_t{point_field_t{weierstrass_b_g2_re}, point_field_t{weierstrass_b_g2_im}};
#endif

  /**
   * [Projective representation](https://hyperelliptic.org/EFD/g1p/auto-shortw-projective.html) of G2 curve.
   */
  typedef Projective<g2_point_field_t, scalar_t, g2_b, g2_generator_x, g2_generator_y> g2_projective_t;
  /**
   * Affine representation of G1 curve.
   */
  typedef Affine<g2_point_field_t> g2_affine_t;
#endif

} // namespace curve_config

#endif