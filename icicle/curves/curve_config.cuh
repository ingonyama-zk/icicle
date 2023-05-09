#pragma once

#include "../primitives/extension_field.cuh"
#include "../primitives/projective.cuh"

#include "bls12_381.cuh"
// #include "bn254.cuh"


typedef Field<fp_config> scalar_field_t;
typedef scalar_field_t scalar_t;
typedef Field<fq_config> point_field_t;
typedef Projective<point_field_t, scalar_field_t, group_generator, weierstrass_b> projective_t;
typedef Affine<point_field_t> affine_t;
typedef ExtensionField<fq_config> g2_point_field_t;
typedef Projective<g2_point_field_t, scalar_field_t, g2_group_generator, g2_weierstrass_b> g2_projective_t;
typedef Affine<g2_point_field_t> g2_affine_t;
