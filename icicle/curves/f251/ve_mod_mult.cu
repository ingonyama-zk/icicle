#ifndef _F251_VEC_MULT
#define _F251_VEC_MULT
#include "../../appUtils/vector_manipulation/ve_mod_mult.cuh"
#include "../../primitives/field.cuh"
#include "../../primitives/projective.cuh"
#include "../../utils/storage.cuh"
#include "curve_config.cuh"
#include <iostream>
#include <stdio.h>

extern "C" int32_t vec_mod_mult_scalar_f251(
  F251::scalar_t* inout, F251::scalar_t* scalar_vec, size_t n_elments, size_t device_id, cudaStream_t stream = 0)
{
  // TODO: use device_id when working with multiple devices
  (void)device_id;
  try {
    // TODO: device_id
    vector_mod_mult<F251::scalar_t, F251::scalar_t>(scalar_vec, inout, inout, n_elments, stream);
    return CUDA_SUCCESS;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}

extern "C" int32_t vec_mod_mult_device_scalar_f251(
  F251::scalar_t* inout, F251::scalar_t* scalar_vec, size_t n_elements, size_t device_id)
{
  try {
    vector_mod_mult_device<F251::scalar_t, F251::scalar_t>(scalar_vec, inout, inout, n_elements);
    return CUDA_SUCCESS;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}

extern "C" int32_t matrix_vec_mod_mult_f251(
  F251::scalar_t* matrix_flattened,
  F251::scalar_t* input,
  F251::scalar_t* output,
  size_t n_elments,
  size_t device_id,
  cudaStream_t stream = 0)
{
  // TODO: use device_id when working with multiple devices
  (void)device_id;
  try {
    // TODO: device_id
    matrix_mod_mult<F251::scalar_t>(matrix_flattened, input, output, n_elments, stream);
    return CUDA_SUCCESS;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}
#endif