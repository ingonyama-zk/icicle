#ifndef _BLS12_377
#define _BLS12_377
#include <stdio.h>
#include <iostream>
#include "../../primitives/field.cuh"
#include "../../utils/storage.cuh"
#include "../../primitives/projective.cuh"
#include "../../curves/curve_config.cuh"
#include "ve_mod_mult.cuh"


extern "C" int32_t vec_mod_mult_point_bls12_377(BLS12_377::projective_t *inout,
                                      BLS12_377::scalar_t *scalar_vec,
                                      size_t n_elments,
                                      size_t device_id)
{
  try
  {
    // TODO: device_id
    vector_mod_mult<BLS12_377::projective_t, BLS12_377::scalar_t>(scalar_vec, inout, inout, n_elments);
    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}

extern "C" int32_t vec_mod_mult_scalar_bls12_377(BLS12_377::scalar_t *inout,
                                       BLS12_377::scalar_t *scalar_vec,
                                       size_t n_elments,
                                       size_t device_id)
{
  try
  {
    // TODO: device_id
    vector_mod_mult<BLS12_377::scalar_t, BLS12_377::scalar_t>(scalar_vec, inout, inout, n_elments);
    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}

extern "C" int32_t matrix_vec_mod_mult_bls12_377(BLS12_377::scalar_t *matrix_flattened,
                                       BLS12_377::scalar_t *input,
                                       BLS12_377::scalar_t *output,
                                       size_t n_elments,
                                       size_t device_id)
{
  try
  {
    // TODO: device_id
    matrix_mod_mult<BLS12_377::scalar_t>(matrix_flattened, input, output, n_elments);
    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}
#endif