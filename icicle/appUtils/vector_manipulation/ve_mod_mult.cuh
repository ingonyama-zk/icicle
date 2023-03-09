#include <stdexcept>
#include <cuda.h>
#include "../../primitives/base_curve.cuh"
#include "../../curves/curve_config.cuh"
#include "../../primitives/projective.cuh"


/// TESTING
// #define VECTOR_SIZE 2048
#define MAX_THREADS_PER_BLOCK 256

template <typename E, typename S>
int vector_mod_mult(S *scalar_vec, E *element_vec, E *result, size_t n_elments);


extern "C" int32_t vec_mod_mult_point(projective_t *inout,
                                      scalar_t *scalar_vec,
                                      size_t n_elments,
                                      size_t device_id)
{
    try
    {
        // TODO: device_id
        vector_mod_mult<projective_t, scalar_t>(scalar_vec, inout, inout, n_elments);
        return CUDA_SUCCESS;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what()); // TODO: error code and message
        // out->z = 0; //TODO: .set_infinity()
        return -1;
    }
}

extern "C" int32_t vec_mod_mult_scalar(scalar_t *inout,
                                       scalar_t *scalar_vec,
                                       size_t n_elments,
                                       size_t device_id)
{
    try
    {
        // TODO: device_id
        vector_mod_mult<scalar_t, scalar_t>(scalar_vec, inout, inout, n_elments);
        return CUDA_SUCCESS;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what()); // TODO: error code and message
        // out->z = 0; //TODO: .set_infinity()
        return -1;
    }
}