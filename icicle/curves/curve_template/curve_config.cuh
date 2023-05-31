#pragma once

#include "../../primitives/field.cuh"
#include "../../primitives/projective.cuh"

#include "params.cuh"

namespace BN254 {
    typedef Field<CURVE_NAME_U::fp_config> scalar_field_t;    
    typedef scalar_field_t scalar_t;    
    typedef Field<CURVE_NAME_U::fq_config> point_field_t;
    typedef Projective<point_field_t, scalar_field_t, CURVE_NAME_U::group_generator, CURVE_NAME_U::weierstrass_b> projective_t;
    typedef Affine<point_field_t> affine_t;
}