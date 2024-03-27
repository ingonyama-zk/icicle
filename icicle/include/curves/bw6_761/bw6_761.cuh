#pragma once
#ifndef BW6_761_H
#define BW6_761_H

#include "curves/projective.cuh"
#if defined(G2)
#include "fields/extension_field.cuh"
#endif

#include "bw6_761_params.cuh"

namespace bw6_761 {
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

#if defined(G2)
  static constexpr g2_point_field_t g2_generator_x =
    g2_point_field_t{point_field_t{g2_gen_x_re}, point_field_t{g2_gen_x_im}};
  static constexpr g2_point_field_t g2_generator_y =
    g2_point_field_t{point_field_t{g2_gen_y_re}, point_field_t{g2_gen_y_im}};
  static constexpr g2_point_field_t g2_b =
    g2_point_field_t{point_field_t{weierstrass_b_g2_re}, point_field_t{weierstrass_b_g2_im}};
  /**
   * [Projective representation](https://hyperelliptic.org/EFD/g1p/auto-shortw-projective.html) of G2 curve.
   */
  typedef Projective<g2_point_field_t, scalar_t, g2_b, g2_generator_x, g2_generator_y> g2_projective_t;
  /**
   * Affine representation of G1 curve.
   */
  typedef Affine<g2_point_field_t> g2_affine_t;
#endif
} // namespace bw6_761

#endif