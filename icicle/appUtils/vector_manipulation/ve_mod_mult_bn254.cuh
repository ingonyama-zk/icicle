#ifndef _BN254
#define _BN254
#include <stdio.h>
#include <iostream>
#include "../../primitives/field.cuh"
#include "../../utils/storage.cuh"
#include "../../primitives/projective.cuh"
#include "../../curves/curve_config.cuh"
#include "ve_mod_mult.cuh"


extern "C" int32_t vec_mod_mult_point_bn254(BN254::projective_t *inout,
                                      BN254::scalar_t *scalar_vec,
                                      size_t n_elments,
                                      size_t device_id)
{
  try
  {
    // TODO: device_id
    vector_mod_mult<BN254::projective_t, BN254::scalar_t>(scalar_vec, inout, inout, n_elments);
    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}

extern "C" int32_t vec_mod_mult_scalar_bn254(BN254::scalar_t *inout,
                                       BN254::scalar_t *scalar_vec,
                                       size_t n_elments,
                                       size_t device_id)
{
  try
  {
    // TODO: device_id
    vector_mod_mult<BN254::scalar_t, BN254::scalar_t>(scalar_vec, inout, inout, n_elments);
    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}

extern "C" int32_t matrix_vec_mod_mult_bn254(BN254::scalar_t *matrix_flattened,
                                       BN254::scalar_t *input,
                                       BN254::scalar_t *output,
                                       size_t n_elments,
                                       size_t device_id)
{
  try
  {
    // TODO: device_id
    matrix_mod_mult<BN254::scalar_t>(matrix_flattened, input, output, n_elments);
    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}
#endif