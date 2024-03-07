#include "curves/curve_config.cuh"
#include "field.cuh"
#include "utils/utils.h"

using namespace curve_config;

extern "C" void CONCAT_EXPAND(CURVE, GenerateScalars)(scalar_t* scalars, int size)
{
  scalar_t::RandHostMany(scalars, size);
}
