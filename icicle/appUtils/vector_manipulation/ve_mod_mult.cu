#include <stdio.h>
#include <iostream>
#include "../../primitives/field.cuh"
#include "../../utils/storage.cuh"
#include "../../primitives/projective.cuh"
#include "../../curves/curve_config.cuh"
#include "ve_mod_mult.cuh"


extern "C" int32_t vec_mod_mult_point(projective_t *inout,
                                      scalar_t *scalar_vec,
                                      size_t n_elments,
                                      size_t device_id)
{
  // TODO: use device_id when working with multiple devices
  (void)device_id;
  try
  {
    // TODO: device_id
    vector_mod_mult<projective_t, scalar_t>(scalar_vec, inout, inout, n_elments);
    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}

extern "C" int32_t vec_mod_mult_scalar(scalar_t *inout,
                                       scalar_t *scalar_vec,
                                       size_t n_elments,
                                       size_t device_id)
{
  // TODO: use device_id when working with multiple devices
  (void)device_id;
  try
  {
    // TODO: device_id
    vector_mod_mult<scalar_t, scalar_t>(scalar_vec, inout, inout, n_elments);
    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}

extern "C" int32_t matrix_vec_mod_mult(scalar_t *matrix_flattened,
                                       scalar_t *input,
                                       scalar_t *output,
                                       size_t n_elments,
                                       size_t device_id)
{
  // TODO: use device_id when working with multiple devices
  (void)device_id;
  try
  {
    // TODO: device_id
    matrix_mod_mult<scalar_t>(matrix_flattened, input, output, n_elments);
    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}
