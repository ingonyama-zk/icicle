#pragma once
#ifndef INDEX_H
#define INDEX_H

#define BN254      1
#define BLS12_381  2
#define BLS12_377  3

#include "../primitives/field.cuh"
#include "../primitives/projective.cuh"
#if defined(G2_DEFINED)
#include "../primitives/extension_field.cuh"
#endif

#if CURVE_ID == BN254
#include "bn254_params.cuh"
using namespace bn254;
#elif CURVE_ID == BLS12_381
#include "bls12_381_params.cuh"
using namespace bls12_381;
#elif CURVE_ID == BLS12_377
#include "bls12_377_params.cuh"
using namespace bls12_377;
#endif

namespace curve_config {

typedef Field<fp_config> scalar_field_t;
typedef scalar_field_t scalar_t;
typedef Field<fq_config> point_field_t;
static constexpr point_field_t b = point_field_t{weierstrass_b};
typedef Projective<point_field_t, scalar_field_t, b> projective_t;
typedef Affine<point_field_t> affine_t;

#if defined(G2_DEFINED)
typedef ExtensionField<fq_config> g2_point_field_t;
static constexpr g2_point_field_t b_g2 = g2_point_field_t{
  point_field_t{weierstrass_b_g2_re}, point_field_t{weierstrass_b_g2_im}};
typedef Projective<g2_point_field_t, scalar_field_t, b_g2> g2_projective_t;
typedef Affine<g2_point_field_t> g2_affine_t;
#endif

} // namespace curve_config

#endif
