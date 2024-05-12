#include "../../../include/fields/id.h"
#define FIELD_ID BN254

#ifdef ECNTT
#define CURVE_ID BN254
#include "../../../include/curves/curve_config.cuh"
typedef field_config::scalar_t test_scalar;
typedef curve_config::projective_t test_data;
#else
#include "../../../include/fields/field_config.cuh"
typedef field_config::scalar_t test_scalar;
typedef field_config::scalar_t test_data;
#endif

#include "../../../include/fields/field.cuh"
#include "../../../include/curves/projective.cuh"
#include <chrono>
#include <iostream>
#include <vector>

#include "../ntt.cpp"
// #include "../kernel_ntt.cpp"
#include <memory>

void random_samples(test_data* res, uint32_t count)
{
  for (int i = 0; i < count; i++)
    res[i] = i < 1000 ? test_data::rand_host() : res[i - 1000];
}

void incremental_values(test_scalar* res, uint32_t count)
{
  for (int i = 0; i < count; i++) {
    res[i] = i ? res[i - 1] + test_scalar::one() : test_scalar::zero();
  }
}

void transpose_batch(test_scalar* in, test_scalar* out, int row_size, int column_size)
{
  return;
}

int main(int argc, char** argv)
{
  return 0;
}