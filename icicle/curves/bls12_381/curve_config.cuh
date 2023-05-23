#pragma once


#include "../../primitives/field.cuh"
#include "../../primitives/projective.cuh"

#include "params.cuh"

namespace BLS12_381 {
    typedef Field<PARAMS_BLS12_381::fp_config> scalar_field_t;
    typedef scalar_field_t scalar_t;
    typedef Field<PARAMS_BLS12_381::fq_config> point_field_t;
    typedef Projective<point_field_t, scalar_field_t, PARAMS_BLS12_381::group_generator, PARAMS_BLS12_381::weierstrass_b> projective_t;
    typedef Affine<point_field_t> affine_t;
}