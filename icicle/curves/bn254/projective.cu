#include <cuda.h>
#include "c_api.h"
#include "curve_config.cuh"
#include "../../primitives/projective.cuh"

extern "C" bool eq_bn254(BN254::projective_t *point1, BN254::projective_t *point2, size_t device_id = 0)
{
    return (*point1 == *point2);
}