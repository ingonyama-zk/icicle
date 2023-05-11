#ifndef _CURVE_NAME_U_VEC_MULT
#define _CURVE_NAME_U_VEC_MULT
#include <stdio.h>
#include <iostream>
#include "../../primitives/field.cuh"
#include "../../utils/storage.cuh"
#include "../../primitives/projective.cuh"
#include "curve_config.cuh"
#include "../../appUtils/vector_manipulation/ve_mod_mult.cuh"


extern "C" int32_t vec_mod_mult_point_CURVE_NAME_L(CURVE_NAME_U::projective_t *inout,
                                      CURVE_NAME_U::scalar_t *scalar_vec,
                                      size_t n_elments,
                                      size_t device_id)
{
  try
  {
    // TODO: device_id
    vector_mod_mult<CURVE_NAME_U::projective_t, CURVE_NAME_U::scalar_t>(scalar_vec, inout, inout, n_elments);
    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}

extern "C" int32_t vec_mod_mult_scalar_CURVE_NAME_L(CURVE_NAME_U::scalar_t *inout,
                                       CURVE_NAME_U::scalar_t *scalar_vec,
                                       size_t n_elments,
                                       size_t device_id)
{
  try
  {
    // TODO: device_id
    vector_mod_mult<CURVE_NAME_U::scalar_t, CURVE_NAME_U::scalar_t>(scalar_vec, inout, inout, n_elments);
    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}

extern "C" int32_t matrix_vec_mod_mult_CURVE_NAME_L(CURVE_NAME_U::scalar_t *matrix_flattened,
                                       CURVE_NAME_U::scalar_t *input,
                                       CURVE_NAME_U::scalar_t *output,
                                       size_t n_elments,
                                       size_t device_id)
{
  try
  {
    // TODO: device_id
    matrix_mod_mult<CURVE_NAME_U::scalar_t>(matrix_flattened, input, output, n_elments);
    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}
#endif