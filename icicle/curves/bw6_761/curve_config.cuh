#pragma once

#include "../../primitives/field.cuh"
#include "../../primitives/projective.cuh"
#if defined(G2_DEFINED)
#undef G2_DEFINED
#include "../bls12_377/params.cuh"
#define G2_DEFINED
#else
#include "../bls12_377/params.cuh"
#endif

#include "params.cuh"

namespace BW6_761 {
  typedef Field<PARAMS_BLS12_377::fq_config> scalar_t;
  typedef Field<PARAMS_BW6_761::fq_config> point_field_t;
  static constexpr point_field_t gen_x = point_field_t{PARAMS_BW6_761::g1_gen_x};
  static constexpr point_field_t gen_y = point_field_t{PARAMS_BW6_761::g1_gen_y};
  static constexpr point_field_t b = point_field_t{PARAMS_BW6_761::weierstrass_b};
  typedef Projective<point_field_t, scalar_t, b, gen_x, gen_y> projective_t;
  typedef Affine<point_field_t> affine_t;
#if defined(G2_DEFINED)
  static constexpr point_field_t g2_gen_x = point_field_t{PARAMS_BW6_761::g2_gen_x};
  static constexpr point_field_t g2_gen_y = point_field_t{PARAMS_BW6_761::g2_gen_y};
  static constexpr point_field_t g2_b = point_field_t{PARAMS_BW6_761::g2_weierstrass_b};
  typedef Projective<point_field_t, scalar_t, g2_b, g2_gen_x, g2_gen_y> g2_projective_t;
  typedef Affine<point_field_t> g2_affine_t;
#endif
} // namespace BW6_761
