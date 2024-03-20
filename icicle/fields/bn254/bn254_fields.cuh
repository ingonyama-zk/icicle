#include "fields/field.cuh"
#include "bn254_params.cuh"

namespace bn254 {
    typedef Field<fp_config> scalar_t;
    typedef Field<fp_config> point_field_t;
}