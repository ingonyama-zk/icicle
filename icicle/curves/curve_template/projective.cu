#include <cuda.h>
#include "curve_config.cuh"
#include "../../primitives/projective.cuh"

extern "C" bool eq_CURVE_NAME_L(CURVE_NAME_U::projective_t *point1, CURVE_NAME_U::projective_t *point2, size_t device_id = 0)
{
    return (*point1 == *point2);
}