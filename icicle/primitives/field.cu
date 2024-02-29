#include "curves/curve_config.cuh"
#include "field.cuh"
#include "utils/utils.h"

using namespace curve_config;

template <>
int scalar_t::seed = 0;
template <>
int point_field_t::seed = 0;

extern "C" void CONCAT_EXPAND(CURVE, GenerateScalars)(scalar_t* scalars, int size)
{
  scalar_t::RandHostMany(scalars, size);
}
