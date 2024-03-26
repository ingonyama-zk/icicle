#include "field.cuh"
#include "utils/utils.h"

extern "C" void CONCAT_EXPAND(FIELD, GenerateRandomFieldElements)(scalar_t* scalars, int size)
{
  scalar_t::RandHostMany(scalars, size);
}

template <>
struct SharedMemory<scalar_t> {
  __device__ scalar_t* getPointer()
  {
    extern __shared__ scalar_t s_scalar_[];
    return s_scalar_;
  }
};

template <>
struct SharedMemory<point_field_t> {
  __device__ point_field_t* getPointer()
  {
    extern __shared__ point_field_t s_point_field_[];
    return s_point_field_;
  }
};