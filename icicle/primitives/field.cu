#include "../curves/curve_config.cuh"
#include "field.cuh"

#define scalar_t curve_config::scalar_t

extern "C" void GenerateScalars(scalar_t* scalars, int size) { scalar_t::RandHostMany(scalars, size); }
