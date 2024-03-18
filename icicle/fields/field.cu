#include "field.cuh"
#include "utils/utils.h"

extern "C" void CONCAT_EXPAND(CURVE, GenerateRandomFieldElements)(Field* scalars, int size)
{
  scalar_t::RandHostMany(scalars, size);
}
