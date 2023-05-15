#include "params.cuh"
namespace BLS12_377 {
    typedef Field<PARAMS_BLS12_377::fp_config> scalar_field_t;    typedef scalar_field_t scalar_t;    typedef Field<PARAMS_BLS12_377::fq_config> point_field_t;
    typedef Projective<point_field_t, scalar_field_t, PARAMS_BLS12_377::group_generator, PARAMS_BLS12_377::weierstrass_b> projective_t;
    typedef Affine<point_field_t> affine_t;
}