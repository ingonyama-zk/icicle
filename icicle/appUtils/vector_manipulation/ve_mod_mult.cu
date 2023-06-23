#include <stdio.h>
#include <iostream>
#include "../../primitives/field.cuh"
#include "../../utils/storage.cuh"
#include "../../primitives/projective.cuh"
#include "../../curves/curve_config.cuh"
#include "ve_mod_mult.cuh"


// extern "C" int32_t vec_mod_mult_point(projective_t *inout,
//                                       scalar_t *scalar_vec,
//                                       size_t n_elments,
//                                       size_t device_id)
// {
//   try
//   {
//     // TODO: device_id
//     vector_mod_mult<projective_t, scalar_t>(scalar_vec, inout, inout, n_elments);
//     return CUDA_SUCCESS;
//   }
//   catch (const std::runtime_error &ex)
//   {
//     printf("error %s", ex.what()); // TODO: error code and message
//     return -1;
//   }
// }

// extern "C" int32_t vec_mod_mult_scalar(scalar_t *inout,
//                                        scalar_t *scalar_vec,
//                                        size_t n_elments,
//                                        size_t device_id)
// {
//   try
//   {
//     // TODO: device_id
//     vector_mod_mult<scalar_t, scalar_t>(scalar_vec, inout, inout, n_elments);
//     return CUDA_SUCCESS;
//   }
//   catch (const std::runtime_error &ex)
//   {
//     printf("error %s", ex.what()); // TODO: error code and message
//     return -1;
//   }
// }

// extern "C" int32_t matrix_vec_mod_mult(scalar_t *matrix_flattened,
//                                        scalar_t *input,
//                                        scalar_t *output,
//                                        size_t n_elments,
//                                        size_t device_id)
// {
//   try
//   {
//     // TODO: device_id
//     matrix_mod_mult<scalar_t>(matrix_flattened, input, output, n_elments);
//     return CUDA_SUCCESS;
//   }
//   catch (const std::runtime_error &ex)
//   {
//     printf("error %s", ex.what()); // TODO: error code and message
//     return -1;
//   }
// }


int main() {
  typedef point_field_t T;
  unsigned N = 1 << 24;
  T *a;
  a = (T*)malloc(sizeof(T) * N);
  T *b;
  b = (T*)malloc(sizeof(T) * N);
  T *c;
  c = (T*)malloc(sizeof(T) * N);
  for (unsigned i = 0; i < (1 << 15); i++) {
    a[i] = T::rand_host();
    b[i] = T::rand_host();
  }
  for (unsigned i = 1; i < (N >> 15); i++) {
    memcpy((void *)(a + (i << 15)), (void *)(a + ((i-1) << 15)), sizeof(T) << 15);
    memcpy((void *)(b + (i << 15)), (void *)(b + ((i-1) << 15)), sizeof(T) << 15);
  }
  // Allocate memory on the device for the input vectors, the output vector, and the modulus
  T *d_vec_a;
  T *d_vec_b, *d_result;
  cudaMalloc(&d_vec_a, N * sizeof(T));
  cudaMalloc(&d_vec_b, N * sizeof(T));
  cudaMalloc(&d_result, N * sizeof(T));

  // Copy the input vectors and the modulus from the host to the device
  cudaMemcpy(d_vec_a, a, N * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vec_b, b, N * sizeof(T), cudaMemcpyHostToDevice);

  // vector_mod_mult<T, T, 1>(d_vec_a, d_vec_b, d_result, N);
  // vector_mod_mult<T, T, 10>(d_vec_a, d_vec_b, d_result, N);
  // vector_mod_mult<T, T, 20>(d_vec_a, d_vec_b, d_result, N);
  // vector_mod_mult<T, T, 30>(d_vec_a, d_vec_b, d_result, N);
  // vector_mod_mult<T, T, 40>(d_vec_a, d_vec_b, d_result, N);
  // vector_mod_mult<T, T, 50>(d_vec_a, d_vec_b, d_result, N);
  // vector_mod_mult<T, T, 60>(d_vec_a, d_vec_b, d_result, N);
  // vector_mod_mult<T, T, 70>(d_vec_a, d_vec_b, d_result, N);
  // vector_mod_mult<T, T, 80>(d_vec_a, d_vec_b, d_result, N);
  // vector_mod_mult<T, T, 90>(d_vec_a, d_vec_b, d_result, N);
  vector_mod_mult<T, T, 100>(d_vec_a, d_vec_b, d_result, N);

  cudaFree(d_vec_a);
  cudaFree(d_vec_b);
  cudaFree(d_result);
}
