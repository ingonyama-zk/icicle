#include <stdexcept>
#include <cuda.h>
#include "../../primitives/base_curve.cuh"
#include "../../curves/curve_config.cuh"
#include "../../primitives/projective.cuh"


#define MAX_THREADS_PER_BLOCK 256

template <typename E, typename S>
int vector_mod_mult(S *scalar_vec, E *element_vec, E *result, size_t n_elments);
