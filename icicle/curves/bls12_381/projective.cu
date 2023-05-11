#include <cuda.h>
#include "curve_config.cuh"
#include "../../primitives/projective.cuh"

extern "C" bool eq_bls12_381(BLS12_381::projective_t *point1, BLS12_381::projective_t *point2, size_t device_id = 0)
{
    return (*point1 == *point2);
}