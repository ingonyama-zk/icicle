#include <cuda.h>
#include "../curves/curve_config.cuh"
#include "projective.cuh"

extern "C" bool eq(projective_t *point1, projective_t *point2, size_t device_id = 0)
{
  return (*point1 == *point2);
}

#if defined(G2_DEFINED)
extern "C" bool eq_g2(g2_projective_t *point1, g2_projective_t *point2, size_t device_id = 0)
{
  return (*point1 == *point2);
}
#endif