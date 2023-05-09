#include <cuda.h>
#include "../curves/curve_config.cuh"
#include "projective.cuh"

extern "C" bool eq_bls12_381(BLS12_381::projective_t *point1, BLS12_381::projective_t *point2, size_t device_id = 0)
{
    return (*point1 == *point2);
}

extern "C" bool eq_bls12_377(BLS12_377::projective_t *point1, BLS12_377::projective_t *point2, size_t device_id = 0)
{
    return (*point1 == *point2);
}

extern "C" bool eq_bn254(BN254::projective_t *point1, BN254::projective_t *point2, size_t device_id = 0)
{
    return (*point1 == *point2);
}