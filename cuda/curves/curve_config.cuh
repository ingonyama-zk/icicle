#pragma once

#include "../primitives/base_curve.cuh"
#include "../primitives/projective.cuh"


#include "bls12_381.cuh"
// #include "bn254.cuh"


typedef Field<fp_config> scalar_field_t;
typedef scalar_field_t scalar_t;
typedef Field<fq_config> point_field_t;
typedef Projective<point_field_t, weierstrass_b> projective_t;
typedef Affine<point_field_t> affine_t;
