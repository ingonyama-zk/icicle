#include "fields/field_config.cuh"

using namespace field_config;

#include "fields/field.cuh"
#include "utils/utils.h"

extern "C" void CONCAT_EXPAND(FIELD, GenerateScalars)(scalar_t* scalars, int size)
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
