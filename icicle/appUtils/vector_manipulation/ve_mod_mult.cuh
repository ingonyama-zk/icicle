#pragma once
#include <stdexcept>
#include <cuda.h>
#include "../../primitives/field.cuh"
#include "../../curves/curve_config.cuh"
#include "../../primitives/projective.cuh"


#define MAX_THREADS_PER_BLOCK 256

template <typename E, typename S>
int vector_mod_mult(S *scalar_vec, E *element_vec, E *result, size_t n_elments);

template <typename E, typename S>
int batch_vector_mult(S *scalar_vec, E *element_vec, size_t n_scalars, size_t batch_size);

template <typename E>
int matrix_mod_mult(E *matrix_elements, E *vector_elements, E *result, size_t dim);
