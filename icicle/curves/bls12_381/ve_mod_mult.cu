#ifndef _BLS12_381_VEC_MULT
#define _BLS12_381_VEC_MULT
#include "../../appUtils/vector_manipulation/ve_mod_mult.cuh"
#include "../../primitives/field.cuh"
#include "../../primitives/projective.cuh"
#include "../../utils/storage.cuh"
#include "curve_config.cuh"
#include <iostream>
#include <stdio.h>

extern "C" int32_t vec_mod_mult_point_bls12_381(
  BLS12_381::projective_t* inout,
  BLS12_381::scalar_t* scalar_vec,
  size_t n_elments,
  size_t device_id,
  cudaStream_t stream = 0)
{
  try {
    // TODO: device_id
    vector_mod_mult<BLS12_381::projective_t, BLS12_381::scalar_t>(scalar_vec, inout, inout, n_elments, stream);
    return CUDA_SUCCESS;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}

extern "C" int32_t vec_mod_mult_scalar_bls12_381(
  BLS12_381::scalar_t* inout,
  BLS12_381::scalar_t* scalar_vec,
  size_t n_elments,
  size_t device_id,
  cudaStream_t stream = 0)
{
  try {
    // TODO: device_id
    vector_mod_mult<BLS12_381::scalar_t, BLS12_381::scalar_t>(scalar_vec, inout, inout, n_elments, stream);
    return CUDA_SUCCESS;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}

extern "C" int32_t matrix_vec_mod_mult_bls12_381(
  BLS12_381::scalar_t* matrix_flattened,
  BLS12_381::scalar_t* input,
  BLS12_381::scalar_t* output,
  size_t n_elments,
  size_t device_id,
  cudaStream_t stream = 0)
{
  try {
    // TODO: device_id
    matrix_mod_mult<BLS12_381::scalar_t>(matrix_flattened, input, output, n_elments, stream);
    return CUDA_SUCCESS;
  } catch (const std::runtime_error& ex) {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}


extern "C" int32_t mult_proj_batch_vec_cuda_bls12_381(BLS12_381::projective_t *inout,
                                      BLS12_381::scalar_t *scalar_vec,
                                      size_t n_scalars,
                                      size_t batch_size,
                                      size_t device_id)
{
  try
  {
    // TODO: device_id
    batch_vector_mult_template(inout, scalar_vec, n_scalars, batch_size);
    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}

extern "C" int32_t mult_sc_batch_vec_cuda_bls12_381(BLS12_381::scalar_t *inout,
                                       BLS12_381::scalar_t *mult_vec,
                                       size_t n_mult,
                                       size_t batch_size,
                                       size_t device_id)
{
  try
  {
    // TODO: device_id
    batch_vector_mult_template(inout, mult_vec, n_mult, batch_size);
    return CUDA_SUCCESS;
  }
  catch (const std::runtime_error &ex)
  {
    printf("error %s", ex.what()); // TODO: error code and message
    return -1;
  }
}
#endif