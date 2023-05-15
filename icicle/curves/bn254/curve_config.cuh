#pragma once

#include "../../primitives/field.cuh"
#include "../../primitives/projective.cuh"

#include "params.cuh"

namespace BN254 {
    typedef Field<PARAMS_BN254::fp_config> scalar_field_t;    
    typedef scalar_field_t scalar_t;    
    typedef Field<PARAMS_BN254::fq_config> point_field_t;
    typedef Projective<point_field_t, scalar_field_t, PARAMS_BN254::group_generator, PARAMS_BN254::weierstrass_b> projective_t;
    typedef Affine<point_field_t> affine_t;
}