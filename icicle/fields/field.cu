#include "field.cuh"
#include "utils/utils.h"

extern "C" void CONCAT_EXPAND(FIELD, GenerateRandomFieldElements)(scalar_t* scalars, int size)
{
  scalar_t::RandHostMany(scalars, size);
}
